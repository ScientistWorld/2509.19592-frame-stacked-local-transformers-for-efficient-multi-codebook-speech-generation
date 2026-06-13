#!/usr/bin/env bash
set -euo pipefail

cd /home/user
BASELINE_STEPS="${BASELINE_STEPS:-500}" AR_STEPS="${AR_STEPS:-5000}" scripts/reproduce.sh
