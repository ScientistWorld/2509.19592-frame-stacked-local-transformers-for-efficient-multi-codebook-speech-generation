#!/usr/bin/env bash
set -euo pipefail

# Estimated download size: 0 bytes. Disk usage after generation: about 16 MB.
# The original Koel-TTS training data and NanoCodec/Koel checkpoints are not
# released with the paper. This script creates the documented synthetic codec
# token replacement dataset used by this gym. The generated dataset is small
# enough to be tracked under data/, and this script regenerates it if missing.

cd "$(dirname "$0")/.."

if [ ! -e data/synthetic_codec/train.pt ]; then
  python data/generate_synthetic.py --out data/synthetic_codec
else
  echo "data/synthetic_codec already exists; skipping"
fi
