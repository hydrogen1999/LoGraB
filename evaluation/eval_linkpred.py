from __future__ import annotations
from pathlib import Path
import yaml
from typing import Set, Tuple, Dict, Any, List

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import random

def _load_pred_edges(path: Path) -> Set[Tuple[int, int]]:
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

def _load_metadata(instance: Path) -> Dict[str, Any]:
    return yaml.safe_load((instance / "metadata.yml").read_text())

def _true_edges(ds_name: str) -> Set[Tuple[int, int]]:
    dataset = Planetoid(str(Path("source_data") / ds_name), name=ds_name)
    data = dataset[0]
    ei = to_undirected(data.edge_index)
    E = set()
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        if u == v: continue
        a, b = (u, v) if u < v else (v, u)
        E.add((a, b))
    return E

def _build_components(E_pred: Set[Tuple[int, int]]) -> Dict[int, int]:
    # return comp_id per node (only nodes that appear in E_pred)
    G = nx.Graph()
    G.add_edges_from(E_pred)
    comp_map = {}
    for cid, comp in enumerate(nx.connected_components(G)):
        for u in comp:
            comp_map[int(u)] = cid
    return comp_map

def _pairs_cross_components(E_true: Set[Tuple[int, int]], comp: Dict[int, int]) -> List[Tuple[int,int]]:
    pos = []
    for (u, v) in E_true:
        cu, cv = comp.get(u, None), comp.get(v, None)
        if cu is None or cv is None:
            continue
        if cu != cv:
            pos.append((u, v))
    return pos

def _sample_negatives(num_nodes: int, size: int, E_true: Set[Tuple[int,int]], comp: Dict[int,int], rng: random.Random):
    neg = set()
    while len(neg) < size:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in E_true:
            continue
        cu, cv = comp.get(u, None), comp.get(v, None)
        if cu is None or cv is None or cu == cv:
            continue
        neg.add((a, b))
    return list(neg)

def eval_linkpred(instance: Path, pred: Path):
    meta = _load_metadata(instance)
    ds_name = meta["dataset"]
    rng = random.Random(42)

    # load data/features
    dataset = Planetoid(str(Path("source_data") / ds_name), name=ds_name)
    data = dataset[0]
    X = data.x.float()
    num_nodes = data.num_nodes

    E_true = _true_edges(ds_name)
    E_pred = _load_pred_edges(pred)
    comp = _build_components(E_pred)

    pos = _pairs_cross_components(E_true, comp)
    if len(pos) == 0:
        print("[linkpred] No positive cross-island edges; AUROC/AP undefined.")
        return

    # cap for very large graphs
    max_pairs = 200_000
    if len(pos) > max_pairs:
        rng.shuffle(pos); pos = pos[:max_pairs]
    neg = _sample_negatives(num_nodes, len(pos), E_true, comp, rng)

    # simple representation-based scorer: feature dot product
    def score(u, v):
        return float((X[u] * X[v]).sum().item())

    y = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
    s = np.array([score(u, v) for (u, v) in pos] + [score(u, v) for (u, v) in neg], dtype=np.float32)

    auroc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    print(f"[linkpred] AUROC {auroc:.4f}  AP {ap:.4f}")