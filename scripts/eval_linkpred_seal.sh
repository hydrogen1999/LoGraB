#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${1:?Usage: $0 <INSTANCE_DIR> [HOPS] [EPOCHS] [BATCH]}"
HOPS="${2:-2}"
EPOCHS="${3:-20}"
BATCH="${4:-64}"

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
if [ -f ".venv/bin/activate" ]; then . .venv/bin/activate; fi

$PY - <<PY "$INSTANCE" "$HOPS" "$EPOCHS" "$BATCH"
from pathlib import Path
import sys
from lograb.evaluation.seal_linkpred import eval_linkpred_seal
eval_linkpred_seal(Path(sys.argv[1]), hops=int(sys.argv[2]), epochs=int(sys.argv[3]), bs=int(sys.argv[4]))
PY