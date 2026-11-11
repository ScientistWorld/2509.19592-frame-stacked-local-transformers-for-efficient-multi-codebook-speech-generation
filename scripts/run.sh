#!/usr/bin/env bash
# Closing check job: add the few-shot-speaker experiment to the rebuilt gym.
# Trains the few-shot reference and comparison models from the final code,
# freezes the fixtures of both tasks, adds the few-shot rows to every saved
# case (variants retrain their few-shot model; shortcuts are rebuilt from
# public data), adds the private-search and few-shot-memorizer cases, scores
# every case through scripts/evaluate_{test,train}.sh, writes the summary,
# restores reference-only scores and validates. Resumable.
# After the job this file is restored to the end-to-end reference reproduction.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python eval/test/hardening/rebuild/run_rebuild.py
