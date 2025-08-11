from pathlib import Path
import gzip, json, yaml
from typing import Set, Tuple

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

from .metrics import coverage_precision_recall_cohesion


def _load_pred_edges(path: Path) -> Set[Tuple[int, int]]:
    """
    Expected format: two integers per line: "u v". Undirected, self-loops ignored.
    """
    E = set()
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            E.add((a, b))
    return E


def _load_instance_nodes(instance: Path) -> Set[int]:
    V = set()
    with gzip.open(instance / "patches.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            V.update(int(x) for x in obj["nodes_global"])
    return V


def _load_metadata(instance: Path):
    return yaml.safe_load((instance / "metadata.yml").read_text())


def _true_edges_for_nodes(ds_name: str, nodes_subset: Set[int]) -> Set[Tuple[int, int]]:
    dataset = Planetoid(str(Path("source_data") / ds_name), name=ds_name)
    data = dataset[0]
    ei = to_undirected(data.edge_index)
    E_true = set()
    V = nodes_subset
    src = ei[0].tolist()
    dst = ei[1].tolist()
    for u, v in zip(src, dst):
        if u == v:
            continue
        if (u in V) and (v in V):
            a, b = (u, v) if u < v else (v, u)
            E_true.add((a, b))
    return E_true


def eval_reconstruct(instance: Path, pred: Path):
    V_true = _load_instance_nodes(instance)
    meta = _load_metadata(instance)
    ds_name = meta["dataset"]

    E_true = _true_edges_for_nodes(ds_name, V_true)
    E_pred = _load_pred_edges(pred)

    cov, prec, rec, coh = coverage_precision_recall_cohesion(E_pred, E_true, V_true)
    print(f"Coverage {cov:.3f}  Precision {prec:.3f}  Recall {rec:.3f}  Cohesion {coh:.3f}")
