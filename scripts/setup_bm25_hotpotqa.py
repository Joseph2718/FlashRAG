#!/usr/bin/env python3
"""
Setup script for running BM25 retrieval on HotpotQA.

This script:
1. Downloads HotpotQA dev split from FlashRAG datasets (HuggingFace)
2. Uses a corpus (general_knowledge or wiki) - download wiki from FlashRAG if needed
3. Builds BM25 index using FlashRAG's framework index builder

Prerequisites: Install FlashRAG with conda faiss per README:
    conda create -n flashrag python=3.10 -y && conda activate flashrag
    conda install -c pytorch faiss-cpu=1.8.0
    pip install -e .

Usage:
    # Full setup with Wikipedia corpus (recommended for HotpotQA):
    python scripts/setup_bm25_hotpotqa.py --use_wiki_corpus --download_dataset

    # Or with custom corpus path:
    python scripts/setup_bm25_hotpotqa.py --data_dir dataset --index_dir indexes --corpus_path path/to/corpus.jsonl --download_dataset

For manual Wikipedia corpus download:
    https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus
Then extract and set --corpus_path to the extracted jsonl file.
"""

import argparse
import os
import subprocess
import sys


def download_wiki_corpus(corpus_dir: str) -> str:
    """Download Wikipedia corpus (wiki18_100w) from HuggingFace FlashRAG datasets."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    import zipfile
    import shutil

    os.makedirs(corpus_dir, exist_ok=True)
    corpus_path = os.path.join(corpus_dir, "wiki18_100w.jsonl")

    if os.path.exists(corpus_path):
        print(f"Wikipedia corpus already exists at {corpus_path}")
        return corpus_path

    print("Downloading wiki18_100w.zip from RUC-NLPIR/FlashRAG_datasets (this may take a few minutes)...")
    try:
        zip_path = hf_hub_download(
            repo_id="RUC-NLPIR/FlashRAG_datasets",
            filename="retrieval-corpus/wiki18_100w.zip",
            repo_type="dataset",
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            jsonl_name = next((n for n in names if n.endswith(".jsonl")), names[0] if names else None)
            if jsonl_name:
                zf.extract(jsonl_name, corpus_dir)
                extracted = os.path.join(corpus_dir, jsonl_name)
                if os.path.normpath(extracted) != os.path.normpath(corpus_path):
                    shutil.move(extracted, corpus_path)
                print(f"Extracted Wikipedia corpus to {corpus_path}")
            else:
                zf.extractall(corpus_dir)
                print(f"Extracted to {corpus_dir}; locate the .jsonl file for --corpus_path")

        return corpus_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Manually download wiki18_100w.zip from:")
        print("  https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/blob/main/retrieval-corpus/wiki18_100w.zip")
        sys.exit(1)


def download_hotpotqa(data_dir: str):
    """Download HotpotQA dev split from HuggingFace FlashRAG datasets."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    hotpotqa_dir = os.path.join(data_dir, "hotpotqa")
    os.makedirs(hotpotqa_dir, exist_ok=True)
    dev_path = os.path.join(hotpotqa_dir, "dev.jsonl")

    if os.path.exists(dev_path):
        print(f"HotpotQA dev.jsonl already exists at {dev_path}")
        return dev_path

    print("Downloading HotpotQA dev.jsonl from RUC-NLPIR/FlashRAG_datasets...")
    try:
        path = hf_hub_download(
            repo_id="RUC-NLPIR/FlashRAG_datasets",
            filename="hotpotqa/dev.jsonl",
            repo_type="dataset",
            local_dir=data_dir,
        )
        print(f"Downloaded to {path}")
        return path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please manually download hotpotqa/dev.jsonl from https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets")
        sys.exit(1)


def build_bm25_index(corpus_path: str, save_dir: str, backend: str = "bm25s"):
    """Build BM25 index using FlashRAG's framework index builder."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    cmd = [
        sys.executable, "-m", "flashrag.retriever.index_builder",
        "--retrieval_method", "bm25",
        "--corpus_path", corpus_path,
        "--save_dir", save_dir,
        "--bm25_backend", backend,
    ]
    print(f"Building BM25 index: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    bm25_index_path = os.path.join(save_dir, "bm25")
    print(f"BM25 index saved to {bm25_index_path}")
    return bm25_index_path


def main():
    parser = argparse.ArgumentParser(description="Setup BM25 + HotpotQA")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory for datasets (HotpotQA will be at data_dir/hotpotqa/)")
    parser.add_argument("--index_dir", type=str, default="indexes",
                        help="Directory to save BM25 index")
    parser.add_argument("--corpus_path", type=str, default=None,
                        help="Path to corpus jsonl. Default: examples/quick_start/indexes/general_knowledge.jsonl")
    parser.add_argument("--download_dataset", action="store_true",
                        help="Download HotpotQA from HuggingFace")
    parser.add_argument("--use_wiki_corpus", action="store_true",
                        help="Download Wikipedia corpus (wiki18_100w) from HuggingFace for HotpotQA")
    parser.add_argument("--skip_index", action="store_true",
                        help="Skip building BM25 index (use existing)")
    parser.add_argument("--bm25_backend", type=str, default="bm25s", choices=["bm25s", "pyserini"])
    args = parser.parse_args()

    corpus_path = args.corpus_path
    if args.use_wiki_corpus:
        corpus_dir = os.path.join(args.index_dir, "corpus")
        corpus_path = download_wiki_corpus(corpus_dir)
    elif corpus_path is None:
        # Use general_knowledge from quick_start as default
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corpus_path = os.path.join(repo_root, "examples", "quick_start", "indexes", "general_knowledge.jsonl")
        if not os.path.exists(corpus_path):
            print(f"Default corpus not found: {corpus_path}")
            print("Use --use_wiki_corpus to download Wikipedia, or provide --corpus_path")
            sys.exit(1)
        print(f"Using default corpus: {corpus_path}")

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    if args.download_dataset:
        download_hotpotqa(args.data_dir)

    if not args.skip_index:
        index_path = build_bm25_index(corpus_path, args.index_dir, args.bm25_backend)
    else:
        index_path = os.path.join(args.index_dir, "bm25")

    print("\n" + "=" * 60)
    print("Setup complete. Run BM25 retrieval on HotpotQA with:")
    print("=" * 60)
    print(f"""
cd examples/methods
python run_exp.py --method_name bm25-naive \\
    --dataset_name hotpotqa \\
    --split dev \\
    --gpu_id 0 \\
    --index_path {os.path.abspath(index_path)} \\
    --corpus_path {os.path.abspath(corpus_path)} \\
    --data_dir {os.path.abspath(args.data_dir)}
""")
    print("Note: Ensure my_config.yaml exists in examples/methods/ with generator paths configured.")


if __name__ == "__main__":
    main()
