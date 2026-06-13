#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
checkpoint_root="${1:-checkpoints}"
pred_dir="${checkpoint_root%/}/predictions"
PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" \
  python eval/test/evaluate.py --pred-dir "$pred_dir" --out scoring/scores_test.json
