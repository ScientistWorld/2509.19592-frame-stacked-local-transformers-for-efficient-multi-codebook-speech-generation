#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
scripts/download.sh

mkdir -p checkpoints/predictions

if [ -e checkpoints/baseline_parallel_1x/model.pt ]; then
  python -m method.predict \
    --data data/downloads/synthetic_codec \
    --checkpoint checkpoints/baseline_parallel_1x/model.pt \
    --out checkpoints/predictions/baseline_parallel_1x.pt
else
  MODEL_NAME=baseline_parallel_1x STACK_FACTOR=1 STEPS="${BASELINE_STEPS:-500}" scripts/baseline.sh
fi

if [ -e checkpoints/ar_lt_2x/model.pt ]; then
  python -m method.predict \
    --data data/downloads/synthetic_codec \
    --checkpoint checkpoints/ar_lt_2x/model.pt \
    --out checkpoints/predictions/ar_lt_2x.pt
else
  MODEL_NAME=ar_lt_2x VARIANT=ar STACK_FACTOR=2 STEPS="${AR_STEPS:-5000}" scripts/method.sh
fi

if [ -e checkpoints/maskgit_lt_2x/model.pt ]; then
  python -m method.predict \
    --data data/downloads/synthetic_codec \
    --checkpoint checkpoints/maskgit_lt_2x/model.pt \
    --out checkpoints/predictions/maskgit_lt_2x.pt
fi

scripts/evaluate.sh
python validate.py --compare
