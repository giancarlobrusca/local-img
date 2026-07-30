#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export HF_XET_HIGH_PERFORMANCE=1
exec python download.py "$@"
