#!/usr/bin/env bash
# Single bundled job for final hardening. Training data/model code is unchanged;
# regenerate larger paired outputs from saved weights, then rescore everything.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m eval.test.hardening.run_checks
cp eval/test/hardening/reference_run.sh scripts/run.sh
