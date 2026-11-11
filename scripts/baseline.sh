#!/usr/bin/env bash
# Train the no-local-transformer parallel baseline and write its predictions.
set -euo pipefail

cd "$(dirname "$0")/.."
scripts/download.sh

MODEL_NAME="${MODEL_NAME:-baseline_parallel_1x}"
STACK_FACTOR="${STACK_FACTOR:-1}"
STEPS="${STEPS:-10000}"
DATA="${DATA:-data/synthetic_codec}"
# As in the paper, the baseline gets the LT models' total layer count in its
# primary decoder (2 primary + 1 LT layer -> 3 primary layers).
MAIN_LAYERS="${MAIN_LAYERS:-3}"

python -m baseline.run_parallel \
  --data "${DATA}" \
  --out "checkpoints/${MODEL_NAME}" \
  --stack-factor "${STACK_FACTOR}" \
  --steps "${STEPS}" \
  --main-layers "${MAIN_LAYERS}"

mkdir -p checkpoints/predictions
python -m method.predict \
  --data "${DATA}" \
  --checkpoint "checkpoints/${MODEL_NAME}/model.pt" \
  --out "checkpoints/predictions/${MODEL_NAME}.pt"
