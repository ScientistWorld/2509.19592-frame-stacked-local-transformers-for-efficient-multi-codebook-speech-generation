#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" \
  python eval/test/evaluate.py --pred-dir checkpoints/predictions --out scoring/scores_test.json
