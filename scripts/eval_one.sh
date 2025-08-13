#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-Cora}"

# Gen params để tìm đúng instance
STRATEGY="${STRATEGY:-d-hop}"
D="${D:-2}"
K="${K:-32}"
SIGMA="${SIGMA:-0.05}"
P="${P:-0.8}"
SEED="${SEED:-42}"
LAP="${LAP:-unnormalized}"
ROOT_DIR="${ROOT_DIR:-instances}"

# Eval params
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
LR="${LR:-1e-2}"
WD="${WD:-5e-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SCORER="${SCORER:-cosine}"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

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

INSTANCE="${ROOT_DIR}/${DATASET}/${TAG}"
if [ ! -d "$INSTANCE" ]; then
  echo "[eval][error] Instance not found: $INSTANCE"
  echo "Hãy chạy scripts/gen_data_one.sh $DATASET trước."
  exit 2
fi

LOG_DIR="artifacts/logs/${DATASET}/${TAG}"
ARTI_DIR="artifacts/${DATASET}/${TAG}"
mkdir -p "$LOG_DIR" "$ARTI_DIR"

# 1) Node classification (lưu embeddings từ SAGE)
for MODEL in sage gcn gat gin; do
  SAVE_EMB_ARG=""
  if [ "$MODEL" = "sage" ]; then
    SAVE_EMB_ARG="--save-embeddings ${ARTI_DIR}/emb_sage.pt"
  fi
  echo "[eval] nodeclf model=$MODEL"
  $PY -m lograb eval \
    --task nodeclf --instance "$INSTANCE" \
    --model "$MODEL" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --wd "$WD" \
    --num-workers "$NUM_WORKERS" $SAVE_EMB_ARG \
    2>&1 | tee "${LOG_DIR}/03_nodeclf_${MODEL}.log"
done

# 2) Dự đoán cạnh baseline từ patches (để có island graph)
PRED="${ARTI_DIR}/pred_from_patches.txt"
scripts/gen_pred_from_patches.sh "$INSTANCE" "$PRED" 2>&1 | tee "${LOG_DIR}/04_pred_from_patches.log"

# 3) Link prediction (dùng embeddings từ SAGE nếu có)
EMB="${ARTI_DIR}/emb_sage.pt"
if [ -f "$EMB" ]; then
  echo "[eval] linkpred (scorer=${SCORER})"
  $PY -m lograb eval --task linkpred --instance "$INSTANCE" --pred "$PRED" \
     --embeddings "$EMB" --scorer "$SCORER" \
     2>&1 | tee "${LOG_DIR}/05_linkpred.log"
else
  echo "[warn] ${EMB} chưa tồn tại (bỏ qua linkpred)."
fi

# 4) Reconstruction
echo "[eval] reconstruct"
$PY -m lograb eval --task reconstruct --instance "$INSTANCE" --pred "$PRED" \
   2>&1 | tee "${LOG_DIR}/06_reconstruct.log"

echo "[eval] done ${DATASET} ✅"
