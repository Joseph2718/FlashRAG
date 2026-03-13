import math
import pickle
import re
from pathlib import Path

import torch

from setwise_toy_utils import build_hotpot_answer_map, enumerate_candidate_sets


LLM_UTILITY_NAME = "llm_precomputed"
LLM_PROMPT_VERSION = "optiset_entropy_v1"
LLM_SCORE_DEFINITION = "u_raw = H0 - H"
LLM_NORMALIZATION_MODE = "per_query_minmax_valid_only"
NO_PASSAGE_PLACEHOLDER = "No evidence passages are provided."


def save_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def model_slug(model_path_or_id):
    text = str(model_path_or_id or "").strip().rstrip("/")
    text = text.split("/")[-1] or "unknown-model"
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    return text or "unknown-model"


def fixed_pair_candidate_family(max_docs, set_size):
    return f"fixed_bm25_pairs_topd{int(max_docs)}_k{int(set_size)}"


def build_llm_utility_metadata(
    *,
    model_path,
    split,
    query_limit,
    max_docs,
    set_size,
    backend="transformers",
    prompt_version=LLM_PROMPT_VERSION,
    score_definition=LLM_SCORE_DEFINITION,
    normalization_mode=LLM_NORMALIZATION_MODE,
    candidate_family=None,
):
    if candidate_family is None:
        candidate_family = fixed_pair_candidate_family(max_docs=max_docs, set_size=set_size)
    return {
        "model_path": str(model_path),
        "model_slug": model_slug(model_path),
        "backend": backend,
        "prompt_version": prompt_version,
        "score_definition": score_definition,
        "normalization_mode": normalization_mode,
        "candidate_family": candidate_family,
        "topD": int(max_docs),
        "set_size": int(set_size),
        "split": str(split),
        "query_limit": query_limit,
    }


def build_llm_utility_cache_filename(split_name, metadata):
    limit_text = "full" if metadata.get("query_limit") is None else f"q{int(metadata['query_limit'])}"
    return (
        f"{split_name}_{metadata['model_slug']}_{metadata['backend']}_{metadata['prompt_version']}_"
        f"entropy_gain_{metadata['normalization_mode']}_topd{metadata['topD']}_"
        f"setk{metadata['set_size']}_{limit_text}.pkl"
    )


def build_llm_utility_cache_path(output_dir, split_name, metadata):
    return str(Path(output_dir) / build_llm_utility_cache_filename(split_name, metadata))


def build_valid_candidate_mask(rank_mask, candidate_sets, *, max_docs=None, device=None):
    rank_mask = torch.as_tensor(rank_mask, dtype=torch.bool)
    if rank_mask.ndim != 2:
        raise ValueError(f"rank_mask must be 2D, got shape={tuple(rank_mask.shape)}")
    if max_docs is not None:
        rank_mask = rank_mask[:, :max_docs]
    candidate_tensor = torch.tensor(candidate_sets, dtype=torch.long)
    valid_counts = rank_mask.sum(dim=1).to(torch.long)
    valid_candidate_mask = torch.zeros(rank_mask.shape[0], len(candidate_sets), dtype=torch.bool)
    for query_idx in range(rank_mask.shape[0]):
        valid_candidate_mask[query_idx] = (candidate_tensor < valid_counts[query_idx]).all(dim=1)
    if device is not None:
        valid_candidate_mask = valid_candidate_mask.to(device)
    return valid_candidate_mask


def summarize_entropy_utility_table(utility_table):
    valid_mask = utility_table["valid_candidate_mask"].detach().cpu()
    summary = {
        "mean_normalized_utility": float(
            utility_table["utilities"][valid_mask].mean().item()
        ) if bool(valid_mask.any()) else 0.0,
        "mean_oracle_utility": float(utility_table["oracle_utilities"].mean().item()),
    }
    if "raw_entropies" in utility_table:
        raw_valid = utility_table["raw_entropies"].detach().cpu()[valid_mask]
        summary["mean_raw_entropy"] = float(raw_valid.mean().item()) if bool(valid_mask.any()) else 0.0
    if "baseline_entropies" in utility_table:
        summary["mean_baseline_entropy"] = float(utility_table["baseline_entropies"].detach().cpu().mean().item())
    if "delta_entropies" in utility_table:
        delta_valid = utility_table["delta_entropies"].detach().cpu()[valid_mask]
        summary["mean_entropy_gain"] = float(delta_valid.mean().item()) if bool(valid_mask.any()) else 0.0
    return summary


def validate_precomputed_utility_table(
    utility_table,
    split_payload,
    *,
    split_name,
    expected_metadata=None,
    max_docs,
    set_size,
):
    required_keys = {
        "utility_name",
        "candidate_sets",
        "utilities",
        "valid_candidate_mask",
        "baseline_set_indices",
        "oracle_utilities",
        "max_docs",
        "set_size",
        "qids",
        "raw_entropies",
        "baseline_entropies",
        "delta_entropies",
        "metadata",
    }
    missing = sorted(required_keys - set(utility_table.keys()))
    if missing:
        raise ValueError(
            f"Precomputed LLM utility table for split={split_name!r} is missing required keys: {missing}"
        )

    if utility_table["utility_name"] != LLM_UTILITY_NAME:
        raise ValueError(
            f"Expected utility_name={LLM_UTILITY_NAME!r}, got {utility_table['utility_name']!r}"
        )
    if int(utility_table["max_docs"]) != int(max_docs):
        raise ValueError(f"Expected max_docs={max_docs}, got {utility_table['max_docs']}")
    if int(utility_table["set_size"]) != int(set_size):
        raise ValueError(f"Expected set_size={set_size}, got {utility_table['set_size']}")

    expected_qids = list(split_payload["qids"])
    if list(utility_table["qids"]) != expected_qids:
        raise ValueError(
            f"Precomputed LLM utility qids do not match split={split_name!r} qids."
        )

    expected_candidate_sets = enumerate_candidate_sets(max_docs=max_docs, set_size=set_size)
    if list(map(tuple, utility_table["candidate_sets"])) != list(map(tuple, expected_candidate_sets)):
        raise ValueError(
            f"Precomputed candidate_sets do not match the fixed BM25 pair family for split={split_name!r}."
        )

    expected_valid_mask = build_valid_candidate_mask(
        split_payload["mask"].detach().cpu(),
        expected_candidate_sets,
        max_docs=max_docs,
    )
    actual_valid_mask = utility_table["valid_candidate_mask"].detach().cpu()
    if expected_valid_mask.shape != actual_valid_mask.shape or not torch.equal(expected_valid_mask, actual_valid_mask):
        raise ValueError(
            f"Precomputed valid_candidate_mask does not match the split mask-derived candidate validity for split={split_name!r}."
        )

    metadata = dict(utility_table["metadata"])
    if expected_metadata is not None:
        mismatches = {}
        for key, expected_value in expected_metadata.items():
            if key not in metadata:
                mismatches[key] = {"expected": expected_value, "actual": "<missing>"}
            elif expected_value != metadata[key]:
                mismatches[key] = {"expected": expected_value, "actual": metadata[key]}
        if mismatches:
            mismatch_lines = [
                f"{key}: expected={value['expected']!r}, actual={value['actual']!r}"
                for key, value in sorted(mismatches.items())
            ]
            raise ValueError(
                "Precomputed LLM utility metadata does not match notebook expectations for "
                f"split={split_name!r}:\n" + "\n".join(mismatch_lines)
            )

    utilities = utility_table["utilities"].detach().cpu()
    if utilities.shape != actual_valid_mask.shape:
        raise ValueError(
            f"utilities shape {tuple(utilities.shape)} does not match valid_candidate_mask shape {tuple(actual_valid_mask.shape)}"
        )
    if not bool(((utilities >= 0.0) & (utilities <= 1.0)).all()):
        raise ValueError("Normalized LLM utilities must stay in [0, 1].")

    baseline = utility_table["baseline_entropies"].detach().cpu()
    raw = utility_table["raw_entropies"].detach().cpu()
    delta = utility_table["delta_entropies"].detach().cpu()
    if baseline.ndim != 1 or baseline.shape[0] != len(expected_qids):
        raise ValueError("baseline_entropies must have shape [n_queries].")
    if raw.shape != utilities.shape or delta.shape != utilities.shape:
        raise ValueError("raw_entropies and delta_entropies must have shape [n_queries, n_candidate_sets].")

    return metadata


def answer_aliases_by_qid_from_payload(query_payload):
    if "golden_answers" in query_payload:
        return {
            str(qid): [str(answer) for answer in answers]
            for qid, answers in query_payload["golden_answers"].items()
        }
    examples = query_payload.get("examples")
    if examples is None:
        raise ValueError("Query payload must contain either golden_answers or examples.")
    return build_hotpot_answer_map(examples)


def truncate_text_to_token_budget(text, tokenizer, max_tokens):
    text = str(text or "")
    if max_tokens is None:
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def build_evidence_block(doc_texts, selected_indices=None, *, tokenizer=None, max_doc_tokens=None):
    if selected_indices is None:
        selected_indices = []
    passages = []
    for passage_rank, doc_idx in enumerate(selected_indices, start=1):
        doc_text = str(doc_texts[doc_idx] or "")
        if tokenizer is not None and max_doc_tokens is not None:
            doc_text = truncate_text_to_token_budget(doc_text, tokenizer, max_doc_tokens)
        passages.append(f"[{passage_rank}] {doc_text}".strip())
    if not passages:
        return NO_PASSAGE_PLACEHOLDER
    return "\n\n".join(passages)


def build_optiset_entropy_prompt(question, doc_texts, selected_indices, *, tokenizer=None, max_doc_tokens=None):
    evidence_block = build_evidence_block(
        doc_texts,
        selected_indices=selected_indices,
        tokenizer=tokenizer,
        max_doc_tokens=max_doc_tokens,
    )
    return (
        "You are given a question and evidence passages. "
        "Answer the question concisely in a few words using only the evidence when possible.\n\n"
        f"Evidence passages:\n{evidence_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def load_causal_lm_for_entropy_scoring(
    model_path,
    *,
    local_files_only=True,
    load_in_4bit=True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = {
        "device_map": "auto",
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return model, tokenizer


def _prepare_prompt_answer_batch(tokenizer, prompts, answers, max_input_tokens):
    batch_input_ids = []
    batch_attention_masks = []
    prompt_lengths = []
    answer_lengths = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for batched entropy scoring.")

    for prompt, answer in zip(prompts, answers):
        answer_ids = tokenizer.encode(str(answer), add_special_tokens=False)
        if not answer_ids:
            raise ValueError(f"Cannot score an empty answer alias: {answer!r}")
        prompt_ids = tokenizer.encode(str(prompt), add_special_tokens=False)
        if max_input_tokens is not None:
            max_prompt_tokens = max(1, max_input_tokens - len(answer_ids))
            prompt_ids = prompt_ids[-max_prompt_tokens:]
        full_ids = prompt_ids + answer_ids
        prompt_lengths.append(len(prompt_ids))
        answer_lengths.append(len(answer_ids))
        batch_input_ids.append(full_ids)

    max_length = max(len(ids) for ids in batch_input_ids)
    for row_idx, input_ids in enumerate(batch_input_ids):
        pad_length = max_length - len(input_ids)
        batch_input_ids[row_idx] = input_ids + [pad_token_id] * pad_length
        batch_attention_masks.append([1] * len(input_ids) + [0] * pad_length)

    return (
        torch.tensor(batch_input_ids, dtype=torch.long),
        torch.tensor(batch_attention_masks, dtype=torch.long),
        prompt_lengths,
        answer_lengths,
    )


def batch_mean_negative_log_prob(model, tokenizer, prompts, answers, *, max_input_tokens):
    input_ids, attention_mask, prompt_lengths, answer_lengths = _prepare_prompt_answer_batch(
        tokenizer,
        prompts,
        answers,
        max_input_tokens=max_input_tokens,
    )

    input_device = next(model.parameters()).device
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :].to(torch.float32)
        shift_labels = input_ids[:, 1:]
        token_log_probs = torch.log_softmax(shift_logits, dim=-1)
        target_log_probs = torch.gather(token_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    entropies = []
    target_log_probs = target_log_probs.detach().cpu()
    for row_idx, (prompt_length, answer_length) in enumerate(zip(prompt_lengths, answer_lengths)):
        start = prompt_length - 1
        end = start + answer_length
        mean_nll = -target_log_probs[row_idx, start:end].mean().item()
        entropies.append(float(mean_nll))
    return entropies


def normalize_delta_entropies(delta_entropies, valid_candidate_mask):
    utilities = torch.zeros_like(delta_entropies, dtype=torch.float32)
    oracle_utilities = torch.zeros(delta_entropies.shape[0], dtype=torch.float32)
    for query_idx in range(delta_entropies.shape[0]):
        valid_idx = valid_candidate_mask[query_idx].nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        valid_values = delta_entropies[query_idx, valid_idx]
        min_value = valid_values.min()
        max_value = valid_values.max()
        if float((max_value - min_value).abs().item()) <= 1e-12:
            normalized = torch.zeros_like(valid_values, dtype=torch.float32)
        else:
            normalized = ((valid_values - min_value) / (max_value - min_value)).to(torch.float32)
        utilities[query_idx, valid_idx] = normalized
        oracle_utilities[query_idx] = normalized.max()
    return utilities, oracle_utilities


def build_entropy_utility_table(
    *,
    qids,
    questions,
    doc_text_rows,
    rank_mask,
    answer_alias_map,
    model,
    tokenizer,
    max_docs,
    set_size,
    batch_size,
    max_doc_tokens,
    max_input_tokens,
    metadata,
):
    candidate_sets = enumerate_candidate_sets(max_docs=max_docs, set_size=set_size)
    valid_candidate_mask = build_valid_candidate_mask(rank_mask, candidate_sets, max_docs=max_docs)
    n_queries = len(qids)
    n_sets = len(candidate_sets)

    raw_entropies = torch.zeros(n_queries, n_sets, dtype=torch.float32)
    baseline_entropies = torch.zeros(n_queries, dtype=torch.float32)
    delta_entropies = torch.zeros(n_queries, n_sets, dtype=torch.float32)
    baseline_set_indices = torch.zeros(n_queries, set_size, dtype=torch.long)

    for query_idx, qid in enumerate(qids):
        answers = answer_alias_map.get(str(qid), [])
        if not answers:
            raise ValueError(f"No golden answers found for qid={qid!r}")

        valid_count = int(torch.as_tensor(rank_mask[query_idx, :max_docs], dtype=torch.bool).sum().item())
        baseline_indices = list(range(min(set_size, valid_count)))
        if len(baseline_indices) < set_size:
            baseline_indices.extend([0] * (set_size - len(baseline_indices)))
        baseline_set_indices[query_idx] = torch.tensor(baseline_indices, dtype=torch.long)

        question = str(questions[str(qid)] if isinstance(questions, dict) else questions[query_idx])
        doc_texts = list(doc_text_rows[query_idx][:max_docs])

        baseline_prompt = build_optiset_entropy_prompt(
            question,
            doc_texts,
            [],
            tokenizer=tokenizer,
            max_doc_tokens=max_doc_tokens,
        )
        baseline_prompts = [baseline_prompt] * len(answers)
        baseline_scores = []
        for batch_start in range(0, len(answers), batch_size):
            batch_end = min(batch_start + batch_size, len(answers))
            baseline_scores.extend(
                batch_mean_negative_log_prob(
                    model,
                    tokenizer,
                    baseline_prompts[batch_start:batch_end],
                    answers[batch_start:batch_end],
                    max_input_tokens=max_input_tokens,
                )
            )
        baseline_entropy = min(baseline_scores)
        baseline_entropies[query_idx] = baseline_entropy

        valid_set_indices = valid_candidate_mask[query_idx].nonzero(as_tuple=False).squeeze(-1).tolist()
        if not valid_set_indices:
            continue

        prompt_batch = []
        answer_batch = []
        set_batch = []
        for set_idx in valid_set_indices:
            candidate = candidate_sets[set_idx]
            prompt = build_optiset_entropy_prompt(
                question,
                doc_texts,
                candidate,
                tokenizer=tokenizer,
                max_doc_tokens=max_doc_tokens,
            )
            for answer in answers:
                prompt_batch.append(prompt)
                answer_batch.append(answer)
                set_batch.append(set_idx)

        set_score_map = {set_idx: [] for set_idx in valid_set_indices}
        for batch_start in range(0, len(prompt_batch), batch_size):
            batch_end = min(batch_start + batch_size, len(prompt_batch))
            batch_scores = batch_mean_negative_log_prob(
                model,
                tokenizer,
                prompt_batch[batch_start:batch_end],
                answer_batch[batch_start:batch_end],
                max_input_tokens=max_input_tokens,
            )
            for local_idx, score in enumerate(batch_scores):
                set_idx = set_batch[batch_start + local_idx]
                set_score_map[set_idx].append(score)

        for set_idx, scores in set_score_map.items():
            entropy = min(scores)
            raw_entropies[query_idx, set_idx] = float(entropy)
            delta_entropies[query_idx, set_idx] = float(baseline_entropy - entropy)

    utilities, oracle_utilities = normalize_delta_entropies(delta_entropies, valid_candidate_mask)
    return {
        "utility_name": LLM_UTILITY_NAME,
        "qids": list(qids),
        "candidate_sets": candidate_sets,
        "utilities": utilities,
        "valid_candidate_mask": valid_candidate_mask,
        "baseline_set_indices": baseline_set_indices,
        "oracle_utilities": oracle_utilities,
        "max_docs": max_docs,
        "set_size": set_size,
        "raw_entropies": raw_entropies,
        "baseline_entropies": baseline_entropies,
        "delta_entropies": delta_entropies,
        "metadata": dict(metadata),
    }
