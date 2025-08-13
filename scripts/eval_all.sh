#!/usr/bin/env bash
set -euo pipefail
DATASETS="${DATASETS:-Cora Citeseer PubMed ogbn-arxiv}"

if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

for ds in $DATASETS; do
  echo "===================="
  echo "[EVAL ALL] Dataset: $ds"
  echo "===================="
  scripts/eval_one.sh "$ds"
done

echo "[EVAL ALL DONE] 🎉"
