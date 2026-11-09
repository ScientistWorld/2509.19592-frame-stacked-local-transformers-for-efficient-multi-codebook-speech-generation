#!/usr/bin/env bash
# End-to-end reference reproduction: the paper's AR local transformer at the
# two scored operating points (2x and 4x frame stacking) plus the comparison
# models (parallel heads at 1x/2x/4x, MaskGIT local transformer at 2x/4x).
# All models use the same training recipe and step count.
set -euo pipefail

cd "$(dirname "$0")/.."
scripts/download.sh
export STEPS="${STEPS:-10000}"

rm -rf checkpoints/predictions
for name in generator_2x generator_4x baseline_maskgit_2x baseline_maskgit_4x \
            baseline_parallel_1x baseline_parallel_2x baseline_parallel_4x; do
  rm -rf "checkpoints/${name}"
done

MODEL_NAME=generator_2x VARIANT=ar STACK_FACTOR=2 scripts/method.sh
MODEL_NAME=generator_4x VARIANT=ar STACK_FACTOR=4 scripts/method.sh
MODEL_NAME=baseline_maskgit_2x VARIANT=maskgit STACK_FACTOR=2 scripts/method.sh
MODEL_NAME=baseline_maskgit_4x VARIANT=maskgit STACK_FACTOR=4 scripts/method.sh
MODEL_NAME=baseline_parallel_1x STACK_FACTOR=1 scripts/baseline.sh
MODEL_NAME=baseline_parallel_2x STACK_FACTOR=2 scripts/baseline.sh
MODEL_NAME=baseline_parallel_4x STACK_FACTOR=4 scripts/baseline.sh
