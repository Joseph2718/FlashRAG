#!/bin/bash
# Run FlashRAG using flashrag_fix env (bypasses conda activate / PATH issues)
# Usage: ./run_flashrag.sh <script> [args...]
#
# Examples:
#   ./run_flashrag.sh scripts/setup_bm25_hotpotqa.py --download_dataset --index_dir indexes
#   ./run_flashrag.sh examples/methods/run_exp.py --method_name bm25-naive --dataset_name hotpotqa --split dev ...

PYTHON="/Users/josephop/miniforge3/envs/flashrag_fix/bin/python"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
exec "$PYTHON" "$@"
