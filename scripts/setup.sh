#!/usr/bin/env bash
set -euo pipefail
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
echo "[setup] done."
