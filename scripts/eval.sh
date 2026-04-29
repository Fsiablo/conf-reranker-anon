#!/usr/bin/env bash
# Evaluate a trained Conf-Reranker checkpoint on a benchmark.
#
# Usage:
#   bash scripts/eval.sh runs/default/final.pt data/dev.jsonl
#
# Reports MRR, nDCG@5, Recall@1/@5, ECE, and selected-set statistics.

set -euo pipefail
CKPT="${1:?usage: eval.sh CKPT EVAL_JSONL}"
EVAL="${2:?usage: eval.sh CKPT EVAL_JSONL}"

python -m scripts.run_eval --ckpt "${CKPT}" --data "${EVAL}"
