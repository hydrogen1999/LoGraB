from pathlib import Path
import gzip, json
from typing import Set, Tuple

from .metrics import coverage_precision_recall_cohesion
from .utils import load_metadata, load_source_graph, load_predicted_edges, true_edges_for_nodes

def _load_instance_nodes(instance: Path) -> Set[int]:
    V = set()
    with gzip.open(instance / "patches.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            V.update(int(x) for x in obj["nodes_global"])
    return V

def eval_reconstruct(instance: Path, pred: Path):
    V_true = _load_instance_nodes(instance)
    meta = load_metadata(instance)
    data = load_source_graph(meta["dataset"])

    E_true = true_edges_for_nodes(data, V_true)
    E_pred = load_predicted_edges(pred)

    cov, prec, rec, coh = coverage_precision_recall_cohesion(E_pred, E_true, V_true)
    print(f"[reconstruct] |V_true|={len(V_true)} |E_true|={len(E_true)} |E_pred|={len(E_pred)}")
    print(f"[reconstruct] Coverage {cov:.3f}  Precision {prec:.3f}  Recall {rec:.3f}  Cohesion {coh:.3f}")