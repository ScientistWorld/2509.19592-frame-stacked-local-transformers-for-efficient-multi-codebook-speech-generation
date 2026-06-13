#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
scripts/download.sh

rm -rf checkpoints/baseline_parallel_1x checkpoints/ar_lt_2x checkpoints/maskgit_lt_2x checkpoints/predictions

MODEL_NAME=baseline_parallel_1x STACK_FACTOR=1 STEPS="${STEPS:-1500}" scripts/baseline.sh
MODEL_NAME=ar_lt_2x VARIANT=ar STACK_FACTOR=2 STEPS="${STEPS:-1500}" scripts/method.sh
MODEL_NAME=maskgit_lt_2x VARIANT=maskgit STACK_FACTOR=2 STEPS="${STEPS:-1500}" scripts/method.sh

python validate.py --compare
