#!/usr/bin/env python3
"""
Standalone BM25 index builder using only bm25s (no FlashRAG imports).

Creates index compatible with FlashRAG's BM25Retriever (bm25s backend).

Usage:
    python scripts/build_bm25_index_standalone.py \
        --corpus_path examples/quick_start/indexes/general_knowledge.jsonl \
        --save_dir indexes
"""

import argparse
import json
import os


def load_corpus_jsonl(corpus_path: str):
    """Load corpus from jsonl. Each line: {"id": ..., "contents": ...} or {"id": ..., "text": ...}"""
    corpus = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "contents" not in item and "text" in item:
                item["contents"] = item["text"]
            corpus.append(item)
    return corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    import bm25s
    import Stemmer

    corpus = load_corpus_jsonl(args.corpus_path)
    if not corpus:
        raise ValueError(f"Empty corpus: {args.corpus_path}")

    # Use datasets-like format for compatibility with BM25Retriever
    corpus_text = [item["contents"] for item in corpus]
    # Convert to format bm25s expects: list of dicts with id, contents
    corpus_for_bm25 = [{"id": i, "contents": c} for i, c in enumerate(corpus_text)]

    save_dir = os.path.join(args.save_dir, "bm25")
    os.makedirs(save_dir, exist_ok=True)

    stemmer = Stemmer.Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=stemmer)

    corpus_tokens = tokenizer.tokenize(corpus_text, return_as="tuple")
    retriever = bm25s.BM25(corpus=corpus_for_bm25, backend="numba")
    retriever.index(corpus_tokens)
    retriever.save(save_dir, corpus=None)
    tokenizer.save_vocab(save_dir)
    tokenizer.save_stopwords(save_dir)

    print(f"BM25 index saved to {save_dir}")


if __name__ == "__main__":
    main()
