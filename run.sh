#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "No .venv found — run ./setup.sh first."
  exit 1
fi

source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_XET_HIGH_PERFORMANCE=1     # faster first-time weight downloads
exec python app.py
