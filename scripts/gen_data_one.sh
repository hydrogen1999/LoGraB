#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-Cora}"

# Params từ env
STRATEGY="${STRATEGY:-d-hop}"
D="${D:-2}"
K="${K:-32}"
SIGMA="${SIGMA:-0.05}"
P="${P:-0.8}"
SEED="${SEED:-42}"
LAP="${LAP:-unnormalized}"
ROOT_DIR="${ROOT_DIR:-instances}"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

# Bật venv nếu có
if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

# Tính TAG như trong code
TAG="$(
"$PY" - <<PY
strategy = "${STRATEGY}"
d = int("${D}")
k = int("${K}")
sigma = float("${SIGMA}")
p = float("${P}")
lap = "${LAP}"
print(f"{strategy}_d{d}_k{k}_s{sigma:.2f}_p{p}_{lap[0]}")
PY
)"

CFG_DIR="artifacts/configs"
LOG_DIR="artifacts/logs/${DATASET}/${TAG}"
mkdir -p "$CFG_DIR" "$LOG_DIR"

CFG="${CFG_DIR}/${DATASET}.yml"
cat > "$CFG" <<EOF
dataset_name: ${DATASET}
root_dir: ${ROOT_DIR}
strategy: ${STRATEGY}
d: ${D}
k: ${K}
sigma: ${SIGMA}
p: ${P}
seed: ${SEED}
laplacian: ${LAP}
EOF

echo "[gen] generate ${DATASET} -> ${CFG}"
$PY -m lograb generate --config "$CFG" 2>&1 | tee "${LOG_DIR}/01_generate.log"

INSTANCE="${ROOT_DIR}/${DATASET}/${TAG}"
echo "[gen] splits -> ${INSTANCE}"
$PY -m lograb splits --instance "$INSTANCE" 2>&1 | tee "${LOG_DIR}/02_splits.log"

echo "[gen] done ${DATASET} (${INSTANCE}) ✅"