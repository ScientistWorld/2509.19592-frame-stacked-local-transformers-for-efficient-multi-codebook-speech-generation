#!/usr/bin/env bash
set -euo pipefail

cd /home/user
scripts/download.sh

if [ ! -e checkpoints/baseline_parallel_1x/model.pt ]; then
  MODEL_NAME=baseline_parallel_1x STACK_FACTOR=1 STEPS=500 scripts/baseline.sh
fi

rm -rf checkpoints/ar_lt_2x checkpoints/predictions
MODEL_NAME=ar_lt_2x VARIANT=ar STACK_FACTOR=2 STEPS="${AR_STEPS:-5000}" scripts/method.sh

mkdir -p checkpoints/predictions
python -m method.predict --data data/downloads/synthetic_codec --checkpoint checkpoints/baseline_parallel_1x/model.pt --out checkpoints/predictions/baseline_parallel_1x.pt
python -m method.predict --data data/downloads/synthetic_codec --checkpoint checkpoints/ar_lt_2x/model.pt --out checkpoints/predictions/ar_lt_2x.pt
scripts/evaluate.sh
python validate.py --compare
