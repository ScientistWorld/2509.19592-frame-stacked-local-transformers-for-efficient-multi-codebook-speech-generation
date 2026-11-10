#!/usr/bin/env bash
# Final hardening check job (temporary entry point; afterwards this file is
# restored to the end-to-end reference reproduction, as at commit 205a717).
# Trains the missing learned variants (20k-step variant at reference seeds
# 17/27, same-budget cosine-LR variant at seeds 7/17/27, the
# train-on-formerly-public-references case), replays every saved case through
# the no-argument hidden test before/after this iteration's fixes, summarizes,
# then restores reference-only scores and validates. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
bash scripts/download.sh
PYTHONPATH="$(pwd)" python eval/test/hardening/final/run_final.py train replay summary restore
