#!/usr/bin/env bash
set -euo pipefail

# Estimated download size: 0 bytes. Disk usage after generation: about 20 MB.
# The original Koel-TTS training data and NanoCodec/Koel checkpoints are not
# released with the paper. This script creates the documented synthetic codec
# token replacement dataset used by this gym.

cd "$(dirname "$0")/.."

if [ ! -e data/downloads/synthetic_codec/train.pt ]; then
  python data/generate_synthetic.py --out data/downloads/synthetic_codec
else
  echo "data/downloads/synthetic_codec already exists; skipping"
fi
