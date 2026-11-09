#!/usr/bin/env bash
# Job entry point: rebuild the gym reference end to end in the container
# (data, reference and comparison models, predictions), freeze the hidden
# test's comparison rows, score both slices and validate the workspace.
set -euo pipefail

cd "$(dirname "$0")/.."
export STEPS="${STEPS:-10000}"

rm -f data/synthetic_codec/*.pt
bash scripts/reproduce.sh
python eval/test/build_fixed_comparators.py
bash scripts/evaluate_test.sh
python validate.py || echo "validate.py reported errors"

# Hardening checks for this iteration: reference seeds, better/weaker
# variants and shortcut reruns through the no-argument hidden test.
# Resumable: finished cases (redteam/hardening/<case>/status.json) are skipped.
rm -rf checkpoints/reference_predictions
mkdir -p checkpoints/reference_predictions
cp checkpoints/predictions/*.pt checkpoints/reference_predictions/
python redteam/hardening/cases.py all
