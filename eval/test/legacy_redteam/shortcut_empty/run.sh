#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/audit.py shortcut_empty
