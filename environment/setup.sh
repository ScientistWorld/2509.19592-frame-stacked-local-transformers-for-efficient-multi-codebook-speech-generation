#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv pip install --system torch==2.12.0 numpy==2.4.6
else
  python -m pip install torch==2.12.0 numpy==2.4.6
fi
