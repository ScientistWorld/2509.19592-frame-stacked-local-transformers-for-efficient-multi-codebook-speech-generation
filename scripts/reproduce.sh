#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
scripts/download.sh

rm -rf checkpoints/baseline_parallel_1x checkpoints/ar_lt_2x checkpoints/ar_lt_4x checkpoints/predictions
[ "${RUN_MASKGIT:-0}" = "1" ] && rm -rf checkpoints/maskgit_lt_2x

MODEL_NAME=baseline_parallel_1x STACK_FACTOR=1 STEPS="${BASELINE_STEPS:-500}" scripts/baseline.sh
MODEL_NAME=ar_lt_2x VARIANT=ar STACK_FACTOR=2 STEPS="${AR_STEPS:-5000}" scripts/method.sh
MODEL_NAME=ar_lt_4x VARIANT=ar STACK_FACTOR=4 STEPS="${AR4_STEPS:-5000}" scripts/method.sh

if [ "${RUN_MASKGIT:-0}" = "1" ]; then
  MODEL_NAME=maskgit_lt_2x VARIANT=maskgit STACK_FACTOR=2 STEPS="${MASKGIT_STEPS:-1500}" scripts/method.sh
fi

python validate.py --compare
