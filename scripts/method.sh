#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="${MODEL_NAME:-ar_lt_2x}"
VARIANT="${VARIANT:-ar}"
STACK_FACTOR="${STACK_FACTOR:-2}"
STEPS="${STEPS:-1500}"

python -m method.train \
  --data data/downloads/synthetic_codec \
  --out "checkpoints/${MODEL_NAME}" \
  --variant "${VARIANT}" \
  --stack-factor "${STACK_FACTOR}" \
  --steps "${STEPS}"

mkdir -p checkpoints/predictions
python -m method.predict \
  --data data/downloads/synthetic_codec \
  --checkpoint "checkpoints/${MODEL_NAME}/model.pt" \
  --out "checkpoints/predictions/${MODEL_NAME}.pt"

scripts/evaluate.sh
