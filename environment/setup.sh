#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv pip install --system torch numpy
else
  python -m pip install torch numpy
fi
