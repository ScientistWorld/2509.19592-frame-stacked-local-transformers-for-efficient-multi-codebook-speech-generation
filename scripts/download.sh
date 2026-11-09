#!/usr/bin/env bash
set -euo pipefail

# Estimated download size: 0 bytes. Disk usage after generation: about 2 MB.
# The paper's Koel-TTS training data and NanoCodec/Koel checkpoints are not
# released. This script creates the documented synthetic codec-token
# replacement dataset (train/validation token splits and the evaluation
# conditioning speakers) with data/generate_synthetic.py. The dataset is small
# and tracked under data/; it is regenerated only if missing.

cd "$(dirname "$0")/.."

if [ ! -e data/synthetic_codec/train.pt ] || [ ! -e data/synthetic_codec/test_conditions.pt ]; then
  python data/generate_synthetic.py --out data/synthetic_codec
else
  echo "data/synthetic_codec already exists; skipping"
fi
