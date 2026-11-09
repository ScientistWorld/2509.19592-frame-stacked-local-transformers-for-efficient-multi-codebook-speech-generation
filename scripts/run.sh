#!/usr/bin/env bash
# Final hardening check job (temporary; restored to the end-to-end reference
# reproduction afterwards, see tmp/run.sh.reference / git history).
# Trains the 20k-step "more training" variant at the two extra reference seeds
# (17, 27) so the better variant is compared with the reference across the same
# three training seeds, then rescores every saved attempt before/after the fix.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
out=eval/test/hardening/raw_predictions
for seed in 17 27; do
  for stack in 2 4; do
    name="better_more_training_seed${seed}"
    ck="checkpoints/variants/${name}/ar_${stack}x"
    target="${out}/${name}/predictions/generator_${stack}x.pt"
    if [ ! -f "${target}" ]; then
      if [ ! -f "${ck}/model.pt" ]; then
        python -m method.train --data data/synthetic_codec --out "${ck}" --variant ar \
          --stack-factor "${stack}" --steps 20000 --seed "${seed}"
      fi
      mkdir -p "$(dirname "${target}")"
      python -m method.predict --data data/synthetic_codec --checkpoint "${ck}/model.pt" \
        --out "${target}.tmp" --seed "${seed}"
      mv "${target}.tmp" "${target}"
    fi
  done
done
PYTHONPATH="$(pwd)" python eval/test/hardening/final/replay.py
