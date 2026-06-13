#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m eval.evaluate_predictions --pred-dir checkpoints/predictions --out scoring/scores.json
