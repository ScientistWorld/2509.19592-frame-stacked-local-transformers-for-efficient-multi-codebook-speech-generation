#!/usr/bin/env bash
set -euo pipefail

cd /home/user
scripts/download.sh

if [ -e checkpoints/baseline_parallel_1x/model.pt ] && \
   [ -e checkpoints/ar_lt_2x/model.pt ] && \
   [ -e checkpoints/maskgit_lt_2x/model.pt ]; then
  rm -rf checkpoints/predictions
  mkdir -p checkpoints/predictions
  python -m method.predict --data data/downloads/synthetic_codec --checkpoint checkpoints/baseline_parallel_1x/model.pt --out checkpoints/predictions/baseline_parallel_1x.pt
  python -m method.predict --data data/downloads/synthetic_codec --checkpoint checkpoints/ar_lt_2x/model.pt --out checkpoints/predictions/ar_lt_2x.pt
  python -m method.predict --data data/downloads/synthetic_codec --checkpoint checkpoints/maskgit_lt_2x/model.pt --out checkpoints/predictions/maskgit_lt_2x.pt
  scripts/evaluate.sh
  python validate.py --compare
else
  STEPS="${STEPS:-500}" scripts/reproduce.sh
fi
