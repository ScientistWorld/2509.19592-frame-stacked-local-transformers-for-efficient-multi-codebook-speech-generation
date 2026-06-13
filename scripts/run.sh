#!/usr/bin/env bash
set -euo pipefail

cd /home/user
STEPS="${STEPS:-500}" scripts/reproduce.sh
