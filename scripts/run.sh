#!/usr/bin/env bash
# Rebuild job (this iteration's single check job): retrain the reference and
# comparison models on the hidden-process data from the final code, freeze the
# evaluator fixtures, train the learned variants, build the shortcut cases,
# score every case through scripts/evaluate_{test,train}.sh, write the
# summary, restore reference-only scores and validate. Resumable.
# After the job this file is restored to the end-to-end reference reproduction.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python eval/test/hardening/rebuild/run_rebuild.py
