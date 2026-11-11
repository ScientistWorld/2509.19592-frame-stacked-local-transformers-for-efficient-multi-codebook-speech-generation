#!/usr/bin/env bash
# End-to-end reference reproduction: train the AR local-transformer reference
# (main task at 2x and 4x frame stacking, few-shot task at 4x) and the
# comparison models (parallel heads at 1x/2x/4x and MaskGIT local transformer
# at 2x/4x on the main task; parallel and MaskGIT at 4x on the few-shot task),
# write their predictions, freeze the evaluator fixtures from them, score both
# slices and validate. The hardening evidence is replayed by
# eval/test/hardening/rebuild/run_rebuild.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
bash scripts/reproduce.sh
python -m eval.test.prepare_fixtures
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh
python validate.py --post-design
