#!/usr/bin/env bash
# Last hardening check job (temporary entry point; afterwards this file is
# restored to the end-to-end reference reproduction).
# Builds the remaining shortcut-family cases (validity-rule best-of-8 filter,
# rule repair, three-piece training splice), replays every saved case through
# the no-argument hidden test and train evaluators, summarizes with a paired
# bootstrap, then restores reference-only scores and validates. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
bash scripts/download.sh
PYTHONPATH="$(pwd)" python eval/test/hardening/last/run_last.py fixtures build replay summary restore
