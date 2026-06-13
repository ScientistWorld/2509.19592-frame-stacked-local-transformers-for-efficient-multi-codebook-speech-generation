#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" \
  python eval/train/evaluate.py --pred-dir checkpoints/predictions --out scoring/scores_train.json
