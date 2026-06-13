#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
scripts/evaluate.sh
bash scripts/evaluate_train.sh checkpoints
bash scripts/evaluate_test.sh checkpoints
python validate.py --compare
