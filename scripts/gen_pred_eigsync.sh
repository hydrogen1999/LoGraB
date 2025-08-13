#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <INSTANCE_DIR> <OUT_PRED_TXT> [K_RECON]"
  exit 1
fi
INSTANCE="$1"
OUT="$2"
K_RECON="${3:-10}"

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
if [ -f ".venv/bin/activate" ]; then . .venv/bin/activate; fi

$PY - <<PY "$INSTANCE" "$OUT" "$K_RECON"
from pathlib import Path
import sys
from lograb.evaluation.reconstruct_eigsync import write_pred_eigsync
write_pred_eigsync(Path(sys.argv[1]), Path(sys.argv[2]), k_recon=int(sys.argv[3]))
PY
