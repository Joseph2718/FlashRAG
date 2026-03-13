#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from setwise_llm_utils import (
    LLM_UTILITY_NAME,
    build_entropy_utility_table,
    build_llm_utility_cache_path,
    build_llm_utility_metadata,
    load_causal_lm_for_entropy_scoring,
    load_pickle,
    save_pickle,
)


DEFAULT_SET_SIZE = 2


def split_payloads_from_setup_cache(setwise_setup_cache):
    return {
        "train": {
            "hotpot_payload": setwise_setup_cache["hotpot_train_payload"],
            "rank_data": setwise_setup_cache["toy_train_rank_data"],
            "option_a_split": setwise_setup_cache["option_a_train"],
        },
        "dev": {
            "hotpot_payload": setwise_setup_cache["hotpot_dev_payload"],
            "rank_data": setwise_setup_cache["toy_dev_rank_data"],
            "option_a_split": setwise_setup_cache["option_a_dev"],
        },
    }


def merge_sharded_tables(shard_tables):
    if not shard_tables:
        raise ValueError("No shard tables were provided for merge.")

    template = shard_tables[0]
    merged = {
        "utility_name": template["utility_name"],
        "qids": [],
        "candidate_sets": template["candidate_sets"],
        "max_docs": template["max_docs"],
        "set_size": template["set_size"],
        "metadata": dict(template["metadata"]),
    }
    tensor_keys = [
        "utilities",
        "valid_candidate_mask",
        "baseline_set_indices",
        "oracle_utilities",
        "raw_entropies",
        "baseline_entropies",
        "delta_entropies",
    ]
    for key in tensor_keys:
        merged[key] = torch.cat([table[key].detach().cpu() for table in shard_tables], dim=0)
    for table in shard_tables:
        merged["qids"].extend(table["qids"])
    return merged


def build_split_slice(split_payload, query_limit):
    option_a_split = split_payload["option_a_split"]
    rank_data = split_payload["rank_data"]
    hotpot_payload = split_payload["hotpot_payload"]

    total_queries = len(option_a_split["qids"])
    if query_limit is None:
        limit = total_queries
    else:
        limit = min(int(query_limit), total_queries)

    qids = list(option_a_split["qids"][:limit])
    return {
        "qids": qids,
        "questions": hotpot_payload["queries"],
        "doc_text_rows": rank_data["doc_texts"][:limit],
        "rank_mask": option_a_split["mask"][:limit],
        "answer_alias_map": hotpot_payload.get("golden_answers"),
        "hotpot_payload": hotpot_payload,
        "max_docs": min(rank_data["mask"].shape[1], len(rank_data["doc_texts"][0]) if rank_data["doc_texts"] else 0),
    }


def shard_cache_path(output_dir, split_name, metadata, shard_start, shard_end):
    filename = (
        build_llm_utility_cache_path(output_dir, split_name, metadata)
        .replace(".pkl", f".shard_{shard_start:05d}_{shard_end:05d}.pkl")
    )
    return filename


def precompute_split(
    *,
    split_name,
    split_payload,
    model,
    tokenizer,
    output_dir,
    query_limit,
    shard_size_queries,
    batch_size,
    max_doc_tokens,
    max_input_tokens,
    model_path,
):
    split_slice = build_split_slice(split_payload, query_limit)
    max_docs = split_slice["max_docs"]
    metadata = build_llm_utility_metadata(
        model_path=model_path,
        split=split_name,
        query_limit=query_limit,
        max_docs=max_docs,
        set_size=DEFAULT_SET_SIZE,
    )
    final_path = build_llm_utility_cache_path(output_dir, split_name, metadata)
    if Path(final_path).exists():
        print(f"[{split_name}] Final utility table already exists: {final_path}")
        return final_path

    answer_alias_map = split_slice["answer_alias_map"]
    if answer_alias_map is None:
        examples = split_slice["hotpot_payload"].get("examples")
        if examples is None:
            raise ValueError(f"[{split_name}] Could not reconstruct golden answers from setup cache payload.")
        from setwise_llm_utils import answer_aliases_by_qid_from_payload

        answer_alias_map = answer_aliases_by_qid_from_payload(split_slice["hotpot_payload"])

    shard_tables = []
    qids = split_slice["qids"]
    for shard_start in range(0, len(qids), shard_size_queries):
        shard_end = min(shard_start + shard_size_queries, len(qids))
        shard_path = shard_cache_path(output_dir, split_name, metadata, shard_start, shard_end)
        if Path(shard_path).exists():
            shard_table = load_pickle(shard_path)
            shard_tables.append(shard_table)
            print(f"[{split_name}] Loaded shard {shard_start}:{shard_end} from {shard_path}")
            continue

        print(f"[{split_name}] Scoring queries {shard_start + 1}-{shard_end} / {len(qids)}")
        shard_table = build_entropy_utility_table(
            qids=qids[shard_start:shard_end],
            questions=split_slice["questions"],
            doc_text_rows=split_slice["doc_text_rows"][shard_start:shard_end],
            rank_mask=split_slice["rank_mask"][shard_start:shard_end],
            answer_alias_map=answer_alias_map,
            model=model,
            tokenizer=tokenizer,
            max_docs=max_docs,
            set_size=DEFAULT_SET_SIZE,
            batch_size=batch_size,
            max_doc_tokens=max_doc_tokens,
            max_input_tokens=max_input_tokens,
            metadata=metadata,
        )
        save_pickle(shard_path, shard_table)
        shard_tables.append(shard_table)
        print(f"[{split_name}] Saved shard to {shard_path}")

    merged_table = merge_sharded_tables(shard_tables)
    save_pickle(final_path, merged_table)
    print(f"[{split_name}] Saved final utility table to {final_path}")
    return final_path


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute OptiSet-inspired LLM entropy utilities over fixed candidate pairs.")
    parser.add_argument("--setwise-setup-cache", required=True, help="Path to the setwise setup cache pickle from rag_frameworks_progress.ipynb")
    parser.add_argument("--split", choices=["train", "dev", "both"], default="both")
    parser.add_argument("--model-path", required=True, help="Local Hugging Face model path")
    parser.add_argument("--output-dir", required=True, help="Directory for shard and final utility pickles")
    parser.add_argument("--query-limit", type=int, default=None, help="Optional query limit applied to the requested split(s)")
    parser.add_argument("--shard-size-queries", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-doc-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_cache = load_pickle(args.setwise_setup_cache)
    split_payload_map = split_payloads_from_setup_cache(setup_cache)

    if args.split == "train" and args.query_limit is None:
        print("[train] query_limit=None will score the full cached train split. This is expensive for an 8B model.")
    if args.split == "both" and args.query_limit is not None:
        print("[both] query_limit applies to both train and dev. Run the script separately per split if you want full dev and capped train.")

    model, tokenizer = load_causal_lm_for_entropy_scoring(
        args.model_path,
        local_files_only=True,
        load_in_4bit=args.load_in_4bit,
    )

    split_names = ["train", "dev"] if args.split == "both" else [args.split]
    final_paths = []
    for split_name in split_names:
        final_paths.append(
            precompute_split(
                split_name=split_name,
                split_payload=split_payload_map[split_name],
                model=model,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
                query_limit=args.query_limit,
                shard_size_queries=args.shard_size_queries,
                batch_size=args.batch_size,
                max_doc_tokens=args.max_doc_tokens,
                max_input_tokens=args.max_input_tokens,
                model_path=args.model_path,
            )
        )

    print("\nFinal utility tables:")
    for path in final_paths:
        print(path)


if __name__ == "__main__":
    main()
