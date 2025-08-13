from __future__ import annotations
from pathlib import Path
from typing import Set, Tuple, Dict, Any, List

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import torch
from .utils import load_metadata, load_source_graph, load_predicted_edges, true_edges_all

def _build_components(E_pred: Set[Tuple[int, int]]) -> Dict[int, int]:
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

def _sample_negatives(num_nodes, size, E_true, comp, rng):
    neg = set()
    nodes_with_comp = list(comp.keys())
    if len(nodes_with_comp) < 2:
        return []
    max_trials = 50 * max(1, size)
    trials = 0
    while len(neg) < size and trials < max_trials:
        u, v = rng.sample(nodes_with_comp, 2)
        trials += 1
        if comp.get(u) == comp.get(v):
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in E_true:
            continue
        neg.add((a, b))
    return list(neg)

def eval_linkpred(instance: Path, pred: Path, embeddings: Path = None, scorer: str = "dot"):
    meta = load_metadata(instance)
    ds_name = meta["dataset"]
    rng = random.Random(42)

    data = load_source_graph(ds_name)
    num_nodes = data.num_nodes

    # Load embeddings if provided, else fall back to data.x (or zeros)
    if embeddings is not None and Path(embeddings).exists():
        X = torch.load(embeddings, map_location='cpu').float()
        if X.ndim == 1:
            X = X.unsqueeze(1)
        if X.size(0) != num_nodes:
            raise ValueError(f"Embeddings rows ({X.size(0)}) != num_nodes ({num_nodes})")
    else:
        X = getattr(data, "x", None)
        if X is None:
            X = torch.zeros((num_nodes, 1), dtype=torch.float32)
        else:
            X = data.x.float()

    # Pre-normalize if cosine (for faster pair scoring)
    if scorer == "cosine":
        Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    else:
        Xn = X

    E_true = true_edges_all(data)
    E_pred = load_predicted_edges(pred)
    comp = _build_components(E_pred)

    pos = _pairs_cross_components(E_true, comp)
    if len(pos) == 0:
        print("[linkpred] No positive cross-island edges; AUROC/AP undefined.")
        return

    max_pairs = 200_000
    if len(pos) > max_pairs:
        rng.shuffle(pos); pos = pos[:max_pairs]
    neg = _sample_negatives(num_nodes, len(pos), E_true, comp, rng)

    def score(u, v):
        return float((Xn[u] * Xn[v]).sum().item())

    y = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
    s = np.array([score(u, v) for (u, v) in pos] + [score(u, v) for (u, v) in neg], dtype=np.float32)

    auroc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    print(f"[linkpred] AUROC {auroc:.4f}  AP {ap:.4f}")