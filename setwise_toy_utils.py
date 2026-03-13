import html
import json
import re
import unicodedata
from itertools import combinations
from pathlib import Path

import torch


def canonicalize_title(title):
    text = html.unescape(str(title or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("_", " ")
    text = text.strip().lower()
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


def load_hotpot_examples(dataset_path):
    dataset_path = Path(dataset_path)
    examples = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            examples.append(json.loads(line))
    return examples


def extract_golden_answers(item):
    answers = item.get("golden_answers")
    if answers is None:
        answers = item.get("answer")

    if answers is None:
        return []
    if isinstance(answers, str):
        answers = [answers]

    deduped_answers = []
    seen = set()
    for answer in answers:
        text = str(answer or "").strip()
        if not text or text in seen:
            continue
        deduped_answers.append(text)
        seen.add(text)
    return deduped_answers


def build_hotpot_answer_map(examples):
    return {
        str(item["id"]): extract_golden_answers(item)
        for item in examples
    }


def load_hotpot_query_payload(dataset_path):
    examples = load_hotpot_examples(dataset_path)
    qids = []
    queries = {}
    support_map = {}
    golden_answers = {}
    for item in examples:
        qid = str(item["id"])
        qids.append(qid)
        queries[qid] = item.get("question", "")
        titles = item.get("metadata", {}).get("supporting_facts", {}).get("title", [])
        support_map[qid] = {canonicalize_title(title) for title in titles if canonicalize_title(title)}
        golden_answers[qid] = extract_golden_answers(item)
    return {
        "examples": examples,
        "qids": qids,
        "queries": queries,
        "support_map": support_map,
        "golden_answers": golden_answers,
    }


def resolve_hotpot_support_dataset_path(split, dataset_dir="dataset/hotpotqa"):
    split = str(split)
    candidate = Path(dataset_dir) / f"{split}.jsonl"
    if candidate.exists():
        return candidate
    available = sorted(path.name for path in Path(dataset_dir).glob("*.jsonl"))
    raise FileNotFoundError(
        f"No HotpotQA support-label file found for split={split!r} at {candidate}. "
        f"Available files: {available}. Download the labeled HotpotQA train/dev files "
        f"before running the toy support-set F1 experiments."
    )


def resolve_hotpot_train_dev_paths(dataset_dir="dataset/hotpotqa"):
    train_path = resolve_hotpot_support_dataset_path("train", dataset_dir=dataset_dir)
    dev_path = resolve_hotpot_support_dataset_path("dev", dataset_dir=dataset_dir)
    return train_path, dev_path


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
    results = torch.zeros(batch_size, n_samples, topk, dtype=torch.float32, device=rankings.device)

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


def _target_device(device, fallback):
    if device is None:
        return fallback
    return torch.device(device)


def _baseline_indices_for_valid_count(valid_count, set_size):
    baseline_indices = list(range(min(set_size, valid_count)))
    if len(baseline_indices) < set_size:
        baseline_indices.extend([0] * (set_size - len(baseline_indices)))
    return baseline_indices


def build_toy_support_utility_table(
    rank_doc_titles,
    rank_gold_titles,
    rank_mask=None,
    *,
    max_docs=10,
    set_size=2,
    candidate_sets=None,
    device=None,
):
    candidate_sets = enumerate_candidate_sets(max_docs=max_docs, set_size=set_size) if candidate_sets is None else list(candidate_sets)
    n_queries = len(rank_doc_titles)
    out_device = _target_device(device, torch.device("cpu"))

    utilities = torch.zeros(n_queries, len(candidate_sets), dtype=torch.float32)
    valid_candidate_mask = torch.zeros(n_queries, len(candidate_sets), dtype=torch.bool)
    baseline_set_indices = torch.zeros(n_queries, set_size, dtype=torch.long)
    oracle_utilities = torch.zeros(n_queries, dtype=torch.float32)

    for query_idx in range(n_queries):
        if rank_mask is None:
            valid_count = min(max_docs, len(rank_doc_titles[query_idx]))
        else:
            valid_count = int(rank_mask[query_idx, :max_docs].detach().sum().item())

        baseline_set_indices[query_idx] = torch.tensor(
            _baseline_indices_for_valid_count(valid_count, set_size),
            dtype=torch.long,
        )

        best_utility = 0.0
        for set_idx, candidate in enumerate(candidate_sets):
            if any(doc_idx >= valid_count for doc_idx in candidate):
                continue
            valid_candidate_mask[query_idx, set_idx] = True
            utility = support_f1_from_indices(
                candidate,
                rank_doc_titles[query_idx],
                rank_gold_titles[query_idx],
            )
            utilities[query_idx, set_idx] = utility
            best_utility = max(best_utility, utility)

        oracle_utilities[query_idx] = best_utility

    return {
        "utility_name": "toy_support_f1",
        "candidate_sets": candidate_sets,
        "utilities": utilities.to(out_device),
        "valid_candidate_mask": valid_candidate_mask.to(out_device),
        "baseline_set_indices": baseline_set_indices.to(out_device),
        "oracle_utilities": oracle_utilities.to(out_device),
        "max_docs": max_docs,
        "set_size": set_size,
    }


def make_utility_provider(
    utility_name,
    utility_table,
    *,
    max_docs,
    set_size,
    candidate_sets=None,
    device=None,
):
    candidate_sets = utility_table.get("candidate_sets") if candidate_sets is None else list(candidate_sets)
    if candidate_sets is None:
        raise ValueError("candidate_sets must be provided to build a utility provider")

    utilities = utility_table["utilities"]
    out_device = _target_device(device, utilities.device)
    utilities = utilities.to(out_device)
    valid_candidate_mask = utility_table["valid_candidate_mask"].to(out_device)
    baseline_set_indices = utility_table["baseline_set_indices"].to(out_device)
    oracle_utilities = utility_table["oracle_utilities"].to(out_device)

    candidate_index = {
        tuple(candidate): idx
        for idx, candidate in enumerate(candidate_sets)
    }
    baseline_choice_rows = []
    baseline_is_valid_rows = []
    for query_idx, indices in enumerate(baseline_set_indices.detach().cpu()):
        candidate_idx = candidate_index.get(tuple(indices.tolist()))
        if candidate_idx is None or not bool(valid_candidate_mask[query_idx, candidate_idx]):
            baseline_choice_rows.append(0)
            baseline_is_valid_rows.append(False)
        else:
            baseline_choice_rows.append(candidate_idx)
            baseline_is_valid_rows.append(True)

    baseline_choice_indices = torch.tensor(
        baseline_choice_rows,
        dtype=torch.long,
        device=out_device,
    )
    baseline_is_valid = torch.tensor(
        baseline_is_valid_rows,
        dtype=torch.bool,
        device=out_device,
    )
    baseline_utilities = torch.zeros(utilities.shape[0], dtype=utilities.dtype, device=out_device)
    if bool(baseline_is_valid.any()):
        valid_query_idx = baseline_is_valid.nonzero(as_tuple=False).squeeze(-1)
        baseline_utilities[valid_query_idx] = utilities[
            valid_query_idx,
            baseline_choice_indices[valid_query_idx],
        ]

    provider = {
        "utility_name": utility_name,
        "candidate_sets": candidate_sets,
        "candidate_index": candidate_index,
        "utilities": utilities,
        "valid_candidate_mask": valid_candidate_mask,
        "baseline_set_indices": baseline_set_indices,
        "baseline_choice_indices": baseline_choice_indices,
        "baseline_is_valid": baseline_is_valid,
        "baseline_utilities": baseline_utilities,
        "oracle_utilities": oracle_utilities,
        "max_docs": max_docs,
        "set_size": set_size,
    }
    for optional_key in ["qids", "raw_entropies", "baseline_entropies", "delta_entropies", "metadata"]:
        if optional_key in utility_table:
            value = utility_table[optional_key]
            if torch.is_tensor(value):
                provider[optional_key] = value.to(out_device)
            else:
                provider[optional_key] = value

    if set_size == 2:
        n_queries = utilities.shape[0]
        pair_lookup = torch.zeros(n_queries, max_docs, max_docs, dtype=utilities.dtype, device=out_device)
        for set_idx, (left_idx, right_idx) in enumerate(candidate_sets):
            pair_lookup[:, left_idx, right_idx] = utilities[:, set_idx]
            pair_lookup[:, right_idx, left_idx] = utilities[:, set_idx]
        provider["pair_lookup"] = pair_lookup

    return provider


def slice_utility_provider(provider, batch_idx):
    batch_idx = batch_idx.to(provider["utilities"].device)
    sliced = {
        "utility_name": provider["utility_name"],
        "candidate_sets": provider["candidate_sets"],
        "candidate_index": provider["candidate_index"],
        "utilities": provider["utilities"][batch_idx],
        "valid_candidate_mask": provider["valid_candidate_mask"][batch_idx],
        "baseline_set_indices": provider["baseline_set_indices"][batch_idx],
        "baseline_choice_indices": provider["baseline_choice_indices"][batch_idx],
        "baseline_is_valid": provider["baseline_is_valid"][batch_idx],
        "baseline_utilities": provider["baseline_utilities"][batch_idx],
        "oracle_utilities": provider["oracle_utilities"][batch_idx],
        "max_docs": provider["max_docs"],
        "set_size": provider["set_size"],
    }
    if "qids" in provider:
        sliced["qids"] = [provider["qids"][idx] for idx in batch_idx.detach().cpu().tolist()]
    for optional_key in ["raw_entropies", "baseline_entropies", "delta_entropies", "metadata"]:
        if optional_key not in provider:
            continue
        value = provider[optional_key]
        if torch.is_tensor(value):
            sliced[optional_key] = value[batch_idx]
        else:
            sliced[optional_key] = value
    if "pair_lookup" in provider:
        sliced["pair_lookup"] = provider["pair_lookup"][batch_idx]
    return sliced


def lookup_set_utility(provider, selected_indices):
    if selected_indices.shape[-1] != provider["set_size"]:
        raise ValueError(
            f"Selected indices last dimension {selected_indices.shape[-1]} does not match provider set_size {provider['set_size']}"
        )

    if provider["set_size"] == 2 and "pair_lookup" in provider:
        query_shape = selected_indices.shape[:-1]
        query_index = torch.arange(selected_indices.shape[0], device=selected_indices.device)
        if selected_indices.ndim > 2:
            view_shape = [selected_indices.shape[0]] + [1] * (selected_indices.ndim - 2)
            query_index = query_index.view(*view_shape).expand(*query_shape)
        left_idx = selected_indices[..., 0]
        right_idx = selected_indices[..., 1]
        return provider["pair_lookup"][query_index, left_idx, right_idx]

    flat_selected = selected_indices.detach().cpu().reshape(-1, provider["set_size"]).tolist()
    query_shape = selected_indices.shape[:-1]
    query_index = torch.arange(selected_indices.shape[0], device=selected_indices.device)
    if selected_indices.ndim > 2:
        view_shape = [selected_indices.shape[0]] + [1] * (selected_indices.ndim - 2)
        query_index = query_index.view(*view_shape).expand(*query_shape)
    flat_query_index = query_index.detach().cpu().reshape(-1).tolist()

    values = []
    utilities_cpu = provider["utilities"].detach().cpu()
    for q_idx, candidate in zip(flat_query_index, flat_selected):
        candidate_key = tuple(sorted(candidate))
        set_idx = provider["candidate_index"].get(candidate_key)
        values.append(0.0 if set_idx is None else float(utilities_cpu[q_idx, set_idx].item()))

    return torch.tensor(values, dtype=provider["utilities"].dtype, device=selected_indices.device).reshape(*query_shape)


def compute_max_set_document_gains(provider):
    gains = torch.zeros(
        provider["utilities"].shape[0],
        provider["max_docs"],
        dtype=provider["utilities"].dtype,
        device=provider["utilities"].device,
    )
    for set_idx, candidate in enumerate(provider["candidate_sets"]):
        candidate_utility = provider["utilities"][:, set_idx]
        for doc_idx in candidate:
            gains[:, doc_idx] = torch.maximum(gains[:, doc_idx], candidate_utility)
    return gains


def compute_mean_set_document_gains(provider):
    gains = torch.zeros(
        provider["utilities"].shape[0],
        provider["max_docs"],
        dtype=provider["utilities"].dtype,
        device=provider["utilities"].device,
    )
    counts = torch.zeros(
        provider["utilities"].shape[0],
        provider["max_docs"],
        dtype=provider["utilities"].dtype,
        device=provider["utilities"].device,
    )
    for set_idx, candidate in enumerate(provider["candidate_sets"]):
        valid_mask = provider["valid_candidate_mask"][:, set_idx].to(provider["utilities"].dtype)
        candidate_utility = provider["utilities"][:, set_idx] * valid_mask
        for doc_idx in candidate:
            gains[:, doc_idx] += candidate_utility
            counts[:, doc_idx] += valid_mask
    return torch.where(counts > 0, gains / counts.clamp_min(1.0), gains)


def _build_set_feature_names(feature_dim):
    extra_feature_names = [
        "set_score_min",
        "set_score_max",
        "set_score_gap",
        "set_lexical_overlap",
        "set_title_match",
    ]
    return (
        [f"mean::{i}" for i in range(feature_dim)]
        + [f"max::{i}" for i in range(feature_dim)]
        + [f"min::{i}" for i in range(feature_dim)]
        + [f"sum::{i}" for i in range(feature_dim)]
        + extra_feature_names
    )


def build_set_feature_data(
    rank_features,
    rank_scores,
    rank_mask,
    rank_doc_titles,
    rank_doc_texts,
    *,
    max_docs=10,
    set_size=2,
    candidate_sets=None,
    chunk_size=256,
    device=None,
    verbose=False,
):
    candidate_sets = enumerate_candidate_sets(max_docs=max_docs, set_size=set_size) if candidate_sets is None else list(candidate_sets)
    candidate_index_tensor = torch.tensor(candidate_sets, dtype=torch.long)

    out_device = _target_device(device, rank_features.device)
    feature_cpu = rank_features[:, :max_docs].detach().to("cpu", dtype=torch.float32)
    score_cpu = rank_scores[:, :max_docs].detach().to("cpu", dtype=torch.float32)
    mask_cpu = rank_mask[:, :max_docs].detach().to("cpu")

    n_queries = feature_cpu.shape[0]
    n_sets = len(candidate_sets)
    feature_dim = feature_cpu.shape[-1]
    set_feature_dim = feature_dim * 4 + 5

    set_features = torch.zeros(n_queries, n_sets, set_feature_dim, dtype=feature_cpu.dtype)
    set_mask = torch.zeros(n_queries, n_sets, dtype=torch.bool)
    baseline_set_indices = torch.zeros(n_queries, set_size, dtype=torch.long)

    for chunk_start in range(0, n_queries, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_queries)
        if verbose:
            print(
                f"Building candidate-set features for queries {chunk_start + 1}-{chunk_end} / {n_queries}"
            )

        for query_idx in range(chunk_start, chunk_end):
            valid_count = int(mask_cpu[query_idx].sum().item())
            baseline_set_indices[query_idx] = torch.tensor(
                _baseline_indices_for_valid_count(valid_count, set_size),
                dtype=torch.long,
            )
            if valid_count <= 0:
                continue

            member_features = feature_cpu[query_idx, candidate_index_tensor]
            member_scores = score_cpu[query_idx, candidate_index_tensor]
            pooled_features = torch.cat(
                [
                    member_features.mean(dim=1),
                    member_features.max(dim=1).values,
                    member_features.min(dim=1).values,
                    member_features.sum(dim=1),
                ],
                dim=-1,
            )

            valid_candidate_mask = (candidate_index_tensor < valid_count).all(dim=1)
            overlaps = torch.zeros(n_sets, dtype=feature_cpu.dtype)
            title_matches = torch.zeros(n_sets, dtype=feature_cpu.dtype)

            for set_idx, candidate in enumerate(candidate_sets):
                if not bool(valid_candidate_mask[set_idx]):
                    continue
                left_idx, right_idx = candidate
                text_a = rank_doc_texts[query_idx][left_idx]
                text_b = rank_doc_texts[query_idx][right_idx]
                title_a = canonicalize_title(rank_doc_titles[query_idx][left_idx])
                title_b = canonicalize_title(rank_doc_titles[query_idx][right_idx])
                overlaps[set_idx] = lexical_overlap_ratio(text_a, text_b)
                title_matches[set_idx] = float(title_a == title_b and title_a != "")

            extra_features = torch.stack(
                [
                    member_scores.min(dim=1).values,
                    member_scores.max(dim=1).values,
                    member_scores.max(dim=1).values - member_scores.min(dim=1).values,
                    overlaps,
                    title_matches,
                ],
                dim=-1,
            )

            all_features = torch.cat([pooled_features, extra_features], dim=-1)
            set_features[query_idx, valid_candidate_mask] = all_features[valid_candidate_mask]
            set_mask[query_idx, valid_candidate_mask] = True

    return {
        "candidate_sets": candidate_sets,
        "features": set_features.to(out_device),
        "mask": set_mask.to(out_device),
        "baseline_set_indices": baseline_set_indices.to(out_device),
        "feature_names": _build_set_feature_names(feature_dim),
    }


def merge_set_feature_and_utility_data(feature_data, utility_provider, qids):
    return {
        "features": feature_data["features"],
        "utilities": utility_provider["utilities"],
        "mask": feature_data["mask"],
        "baseline_choice_indices": utility_provider["baseline_choice_indices"],
        "baseline_utilities": utility_provider["baseline_utilities"],
        "oracle_utilities": utility_provider["oracle_utilities"],
        "utility_provider": utility_provider,
        "candidate_sets": utility_provider["candidate_sets"],
        "feature_names": feature_data["feature_names"],
        "qids": list(qids),
    }


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
    candidate_sets=None,
    chunk_size=256,
    utility_name="toy_support_f1",
):
    if utility_name != "toy_support_f1":
        raise NotImplementedError(
            "Only the toy support-F1 utility is implemented in build_set_candidate_data(). "
            "Use a precomputed utility table with make_utility_provider() for other utilities."
        )

    candidate_sets = enumerate_candidate_sets(max_docs=max_docs, set_size=set_size) if candidate_sets is None else list(candidate_sets)
    utility_table = build_toy_support_utility_table(
        rank_doc_titles,
        rank_gold_titles,
        rank_mask=rank_mask,
        max_docs=max_docs,
        set_size=set_size,
        candidate_sets=candidate_sets,
        device=rank_features.device,
    )
    utility_provider = make_utility_provider(
        utility_name,
        utility_table,
        max_docs=max_docs,
        set_size=set_size,
        candidate_sets=candidate_sets,
        device=rank_features.device,
    )
    feature_data = build_set_feature_data(
        rank_features,
        rank_scores,
        rank_mask,
        rank_doc_titles,
        rank_doc_texts,
        max_docs=max_docs,
        set_size=set_size,
        candidate_sets=candidate_sets,
        chunk_size=chunk_size,
        device=rank_features.device,
    )
    return {
        "candidate_sets": candidate_sets,
        "features": feature_data["features"],
        "utilities": utility_provider["utilities"],
        "mask": feature_data["mask"],
        "baseline_set_indices": utility_provider["baseline_set_indices"],
        "oracle_utilities": utility_provider["oracle_utilities"],
        "feature_names": feature_data["feature_names"],
    }


def mean_support_f1_for_indices(index_rows, doc_title_rows, gold_title_rows):
    scores = [
        support_f1_from_indices(indices, doc_titles, gold_titles)
        for indices, doc_titles, gold_titles in zip(index_rows, doc_title_rows, gold_title_rows)
    ]
    return float(sum(scores) / max(len(scores), 1))
