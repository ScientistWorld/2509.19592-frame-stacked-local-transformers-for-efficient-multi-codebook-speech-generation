#!/usr/bin/env bash
# Train one local-transformer model and write its evaluation predictions.
set -euo pipefail

cd "$(dirname "$0")/.."
scripts/download.sh

MODEL_NAME="${MODEL_NAME:-generator_2x}"
VARIANT="${VARIANT:-ar}"
STACK_FACTOR="${STACK_FACTOR:-2}"
STEPS="${STEPS:-10000}"
DATA="${DATA:-data/synthetic_codec}"

python -m method.train \
  --data "${DATA}" \
  --out "checkpoints/${MODEL_NAME}" \
  --variant "${VARIANT}" \
  --stack-factor "${STACK_FACTOR}" \
  --steps "${STEPS}"

mkdir -p checkpoints/predictions
python -m method.predict \
  --data "${DATA}" \
  --checkpoint "checkpoints/${MODEL_NAME}/model.pt" \
  --out "checkpoints/predictions/${MODEL_NAME}.pt"
