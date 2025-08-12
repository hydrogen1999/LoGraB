from __future__ import annotations
from pathlib import Path
import yaml, gzip, json
from typing import Dict, Any, Set, Tuple
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

def load_metadata(instance_path: Path) -> Dict[str, Any]:
    return yaml.safe_load((instance_path / "metadata.yml").read_text())

def load_source_graph(dataset_name: str):
    return Planetoid(str(Path("source_data") / dataset_name), name=dataset_name)[0]

def load_predicted_edges(pred_path: Path) -> Set[Tuple[int, int]]:
    E = set()
    with pred_path.open() as f:
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

def true_edges_for_nodes(data, nodes_subset: Set[int]) -> Set[Tuple[int, int]]:
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

def true_edges_all(data) -> Set[Tuple[int, int]]:
    ei = to_undirected(data.edge_index)
    E = set()
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        if u == v: continue
        a, b = (u, v) if u < v else (v, u)
        E.add((a, b))
    return E