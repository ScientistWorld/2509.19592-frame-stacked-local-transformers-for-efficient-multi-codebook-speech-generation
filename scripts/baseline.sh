#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
scripts/download.sh

MODEL_NAME="${MODEL_NAME:-baseline_parallel_1x}"
STACK_FACTOR="${STACK_FACTOR:-1}"
STEPS="${STEPS:-1500}"

python -m baseline.run_parallel \
  --data data/synthetic_codec \
  --out "checkpoints/${MODEL_NAME}" \
  --stack-factor "${STACK_FACTOR}" \
  --steps "${STEPS}"

mkdir -p checkpoints/predictions
python -m method.predict \
  --data data/synthetic_codec \
  --checkpoint "checkpoints/${MODEL_NAME}/model.pt" \
  --out "checkpoints/predictions/${MODEL_NAME}.pt"

scripts/evaluate.sh
