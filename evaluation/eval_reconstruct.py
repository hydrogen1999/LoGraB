from pathlib import Path
import gzip, json
from typing import Tuple

from .metrics import coverage_precision_recall_cohesion


def load_edges(path: Path):
    with path.open() as f:
        return {(int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
                for line in f for u, v in [line.strip().split()[:2]]}


def eval_reconstruct(instance: Path, pred: Path):
    # load ground truth adjacency
    with gzip.open(instance / "patches.jsonl.gz", "rt", encoding="utf-8") as f:
        V = set()
        for line in f:
            obj = json.loads(line)
            V.update(obj["nodes_global"])
    # For demo purposes, pretend E_true = empty set (replace with real loader)
    E_true = set()  # TODO: load from hidden GT for camera‑ready
    E_pred = load_edges(pred)
    cov, prec, rec, coh = coverage_precision_recall_cohesion(E_pred, E_true, V)
    print(f"Coverage {cov:.3f}  Precision {prec:.3f}  Recall {rec:.3f}  Cohesion {coh:.3f}")
