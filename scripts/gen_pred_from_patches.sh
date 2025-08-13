#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <INSTANCE_DIR> <OUT_PRED_TXT>"
  exit 1
fi

INSTANCE="$1"
OUT="$2"
mkdir -p "$(dirname "$OUT")"

python - <<'PY'
import gzip, json, sys, yaml
from pathlib import Path
from collections import defaultdict
from lograb.evaluation.utils import load_source_graph

instance = Path(sys.argv[1])
out = Path(sys.argv[2])

# Đọc metadata.yml (YAML, không phải JSON)
meta = yaml.safe_load((instance / "metadata.yml").read_text())
ds_name = meta["dataset"]

# Tải đồ thị gốc một lần
data = load_source_graph(ds_name)
ei = data.edge_index

# Xây dựng adjacency (vô hướng)
adj = defaultdict(set)
src = ei[0].tolist()
dst = ei[1].tolist()
for u, v in zip(src, dst):
    if u == v: 
        continue
    adj[u].add(v)
    adj[v].add(u)

E_pred = set()
with gzip.open(instance / "patches.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        V = set(int(x) for x in row.get("nodes_global", []))
        # Thêm toàn bộ cạnh thật nằm trong patch
        for u in V:
            for v in adj[u]:
                if v in V and u < v:
                    E_pred.add((u, v))

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as fo:
    for u, v in sorted(E_pred):
        fo.write(f"{u} {v}\n")
print(f"[pred] wrote {out} with {len(E_pred)} edges")
PY
"$INSTANCE" "$OUT"