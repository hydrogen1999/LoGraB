# scripts/eval_linkpred_seal.sh
#!/usr/bin/env bash
set -euo pipefail
INSTANCE="${1:?Usage: $0 <INSTANCE_DIR> [HOPS] [EPOCHS] [BATCH] [READOUT]}"
HOPS="${2:-2}"; EPOCHS="${3:-20}"; BATCH="${4:-64}"
READOUT="${5:-max}"                # max | add
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEAL_SEED:-42}"
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
[ -f ".venv/bin/activate" ] && . .venv/bin/activate
$PY - <<PY "$INSTANCE" "$HOPS" "$EPOCHS" "$BATCH" "$READOUT" "$NUM_WORKERS" "$SEED"
from pathlib import Path; import sys
from lograb.evaluation.seal_linkpred import eval_linkpred_seal
eval_linkpred_seal(Path(sys.argv[1]), hops=int(sys.argv[2]), epochs=int(sys.argv[3]),
                   bs=int(sys.argv[4]), readout=sys.argv[5],
                   num_workers=int(sys.argv[6]), seed=int(sys.argv[7]))
PY