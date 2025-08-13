#!/usr/bin/env bash
set -euo pipefail
DATASETS="${DATASETS:-Cora Citeseer PubMed ogbn-arxiv}"

if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

for ds in $DATASETS; do
  echo "===================="
  echo "[GEN ALL] Dataset: $ds"
  echo "===================="
  scripts/gen_data_one.sh "$ds"
done

echo "[GEN ALL DONE] 🎉"
