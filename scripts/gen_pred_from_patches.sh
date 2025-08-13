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
import gzip, json, sys
from pathlib import Path

instance = Path(sys.argv[1])
out = Path(sys.argv[2])

E = set()
with gzip.open(instance / "patches.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        nodes = [int(x) for x in row.get("nodes_global", [])]
        for i in range(len(nodes)-1):
            u, v = nodes[i], nodes[i+1]
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            E.add((a, b))

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as fo:
    for u, v in sorted(E):
        fo.write(f"{u} {v}\\n")
print(f"[pred] wrote {out} with {len(E)} edges")
PY
"$INSTANCE" "$OUT"
