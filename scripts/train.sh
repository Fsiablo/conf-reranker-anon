#!/usr/bin/env bash
# Train Conf-Reranker on a listwise JSONL dataset.
#
# Usage:
#   bash scripts/train.sh configs/default.yaml
#
# Requires GPU (CUDA). For CPU smoke-tests, see `python -m demo`.

set -euo pipefail
CFG="${1:-configs/default.yaml}"

python -m scripts.run_train --config "${CFG}"
