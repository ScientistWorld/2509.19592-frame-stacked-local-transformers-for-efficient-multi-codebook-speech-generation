#!/usr/bin/env bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python redteam/audit.py all 2>&1 | tee redteam/job.log
python redteam/analyze.py | tee redteam/analysis.log
