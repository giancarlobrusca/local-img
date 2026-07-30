#!/usr/bin/env bash
# One-time setup. PyTorch has no wheels for Python 3.14 yet, so we pin 3.12.
set -euo pipefail
cd "$(dirname "$0")"

PY=$(command -v python3.12 || command -v python3.11 || true)
if [ -z "$PY" ]; then
  echo "Need Python 3.11 or 3.12 (PyTorch has no 3.14 wheels)."
  echo "Install with:  brew install python@3.12"
  exit 1
fi

echo "==> venv with $($PY -V)"
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel >/dev/null
echo "==> installing dependencies (a few minutes, ~2.5 GB)"
pip install -r requirements.txt

python - <<'PY'
import torch
print(f"\n==> torch {torch.__version__} | MPS available: {torch.backends.mps.is_available()}")
PY

echo
echo "Done. Start the app with:  ./run.sh"
