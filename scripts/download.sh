#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Synthetic train/val/meta artifacts are tracked original gym data, not external
# checkpoints. The only external pre-existing artifact used is the paper.
if [ ! -s paper/2509.19592.pdf ]; then
  mkdir -p paper
  curl --fail --location --retry 3 https://arxiv.org/pdf/2509.19592 -o paper/2509.19592.pdf
fi
for artifact in train.pt val.pt meta.pt; do
  test -f "data/synthetic_codec/$artifact" || { echo "Missing tracked data artifact: $artifact" >&2; exit 1; }
done
# Speaker inputs only; no test labels or public reference-token sampler.
python data/generate_synthetic.py --out data/synthetic_codec
