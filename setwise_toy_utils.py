import json
import re
from itertools import combinations
from pathlib import Path

import torch


def canonicalize_title(title):
    text = str(title or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_hotpot_support_title_map(dataset_path):
    dataset_path = Path(dataset_path)
    support_map = {}
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            titles = item.get("metadata", {}).get("supporting_facts", {}).get("title", [])
            support_map[item["id"]] = {canonicalize_title(title) for title in titles if canonicalize_title(title)}
    return support_map


def build_rank_doc_views(beir_qids, flashrag_docs, rank_qids, top_m):
    qid_to_docs = {
        qid: list(docs[:top_m])
        for qid, docs in zip(beir_qids, flashrag_docs)
    }
    rank_doc_titles = []
    rank_doc_texts = []
    for qid in rank_qids:
        docs = qid_to_docs[qid]
        titles = []
        texts = []
        for doc in docs:
            title = str(doc.get("title", "") or "")
            contents = str(doc.get("contents", doc.get("text", "")) or "")
            titles.append(title)
            texts.append(f"{title}\n{contents}".strip())
        pad = top_m - len(docs)
        if pad > 0:
            titles.extend([""] * pad)
            texts.extend([""] * pad)
        rank_doc_titles.append(titles[:top_m])
        rank_doc_texts.append(texts[:top_m])
    return rank_doc_titles, rank_doc_texts


def support_set_f1(selected_titles, gold_titles):
    pred = {canonicalize_title(title) for title in selected_titles if canonicalize_title(title)}
    gold = {canonicalize_title(title) for title in gold_titles if canonicalize_title(title)}

    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    overlap = len(pred & gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return float(2.0 * precision * recall / (precision + recall))


def sampled_prefix_support_f1(rankings, batch_doc_titles, batch_gold_titles, topk):
    rankings_cpu = rankings.detach().cpu().tolist()
    batch_size = len(rankings_cpu)
    n_samples = len(rankings_cpu[0]) if batch_size else 0
    results = torch.zeros(batch_size, n_samples, topk, dtype=torch.float64, device=rankings.device)

    for batch_idx in range(batch_size):
        gold_titles = batch_gold_titles[batch_idx]
        query_titles = batch_doc_titles[batch_idx]
        for sample_idx, ranking in enumerate(rankings_cpu[batch_idx]):
            selected_titles = []
            for pos in range(topk):
                selected_titles.append(query_titles[ranking[pos]])
                results[batch_idx, sample_idx, pos] = support_set_f1(selected_titles, gold_titles)
    return results


def support_f1_from_indices(selected_indices, doc_titles, gold_titles):
    selected_titles = [doc_titles[idx] for idx in selected_indices]
    return support_set_f1(selected_titles, gold_titles)


def lexical_overlap_ratio(text_a, text_b):
    tokens_a = set(re.findall(r"[a-z0-9]+", str(text_a).lower()))
    tokens_b = set(re.findall(r"[a-z0-9]+", str(text_b).lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def enumerate_candidate_sets(max_docs, set_size):
    return list(combinations(range(max_docs), set_size))


def build_set_candidate_data(
    rank_features,
    rank_scores,
    rank_mask,
    rank_doc_titles,
    rank_doc_texts,
    rank_gold_titles,
    *,
    max_docs=10,
    set_size=2,
):
    candidate_sets = enumerate_candidate_sets(max_docs=max_docs, set_size=set_size)
    n_queries = rank_features.shape[0]
    feature_dim = rank_features.shape[-1]
    pair_feature_dim = feature_dim * 4 + 5

    set_features = torch.zeros(
        n_queries,
        len(candidate_sets),
        pair_feature_dim,
        dtype=rank_features.dtype,
        device=rank_features.device,
    )
    set_utilities = torch.zeros(
        n_queries,
        len(candidate_sets),
        dtype=torch.float64,
        device=rank_features.device,
    )
    set_mask = torch.zeros(
        n_queries,
        len(candidate_sets),
        dtype=torch.bool,
        device=rank_features.device,
    )
    baseline_set_indices = torch.zeros(
        n_queries,
        set_size,
        dtype=torch.long,
        device=rank_features.device,
    )
    oracle_utilities = torch.zeros(
        n_queries,
        dtype=torch.float64,
        device=rank_features.device,
    )

    extra_feature_names = [
        "set_score_min",
        "set_score_max",
        "set_score_gap",
        "set_lexical_overlap",
        "set_title_match",
    ]

    for query_idx in range(n_queries):
        valid_count = int(rank_mask[query_idx, :max_docs].sum().item())
        baseline_indices = list(range(min(set_size, valid_count)))
        if len(baseline_indices) < set_size:
            baseline_indices.extend([0] * (set_size - len(baseline_indices)))
        baseline_set_indices[query_idx] = torch.tensor(
            baseline_indices,
            dtype=torch.long,
            device=rank_features.device,
        )

        best_utility = 0.0
        for set_idx, candidate in enumerate(candidate_sets):
            if any(doc_idx >= valid_count for doc_idx in candidate):
                continue

            candidate_list = list(candidate)
            member_features = rank_features[query_idx, candidate_list]
            member_scores = rank_scores[query_idx, candidate_list]
            score_min = member_scores.min().item()
            score_max = member_scores.max().item()
            score_gap = score_max - score_min

            text_a = rank_doc_texts[query_idx][candidate_list[0]]
            text_b = rank_doc_texts[query_idx][candidate_list[1]]
            title_a = canonicalize_title(rank_doc_titles[query_idx][candidate_list[0]])
            title_b = canonicalize_title(rank_doc_titles[query_idx][candidate_list[1]])

            overlap = lexical_overlap_ratio(text_a, text_b)
            title_match = float(title_a == title_b and title_a != "")

            extra_features = torch.tensor(
                [score_min, score_max, score_gap, overlap, title_match],
                dtype=rank_features.dtype,
                device=rank_features.device,
            )
            pooled = torch.cat(
                [
                    member_features.mean(dim=0),
                    member_features.max(dim=0).values,
                    member_features.min(dim=0).values,
                    member_features.sum(dim=0),
                    extra_features,
                ],
                dim=0,
            )
            set_features[query_idx, set_idx] = pooled

            utility = support_f1_from_indices(
                candidate_list,
                rank_doc_titles[query_idx],
                rank_gold_titles[query_idx],
            )
            set_utilities[query_idx, set_idx] = utility
            set_mask[query_idx, set_idx] = True
            best_utility = max(best_utility, utility)

        oracle_utilities[query_idx] = best_utility

    set_feature_names = (
        [f"mean::{i}" for i in range(feature_dim)]
        + [f"max::{i}" for i in range(feature_dim)]
        + [f"min::{i}" for i in range(feature_dim)]
        + [f"sum::{i}" for i in range(feature_dim)]
        + extra_feature_names
    )

    return {
        "candidate_sets": candidate_sets,
        "features": set_features,
        "utilities": set_utilities,
        "mask": set_mask,
        "baseline_set_indices": baseline_set_indices,
        "oracle_utilities": oracle_utilities,
        "feature_names": set_feature_names,
    }


def mean_support_f1_for_indices(index_rows, doc_title_rows, gold_title_rows):
    scores = [
        support_f1_from_indices(indices, doc_titles, gold_titles)
        for indices, doc_titles, gold_titles in zip(index_rows, doc_title_rows, gold_title_rows)
    ]
    return float(sum(scores) / max(len(scores), 1))
