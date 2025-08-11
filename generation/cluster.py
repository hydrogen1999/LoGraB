from typing import Dict, List, Any
import torch
import networkx as nx
import metis
from torch_geometric.utils import to_networkx


def partition_metis(data, parts: int = 64) -> Dict[int, int]:
    """Return mapping node ‑> cluster id via METIS."""
    g_nx = to_networkx(data, to_undirected=True)
    _, part_vec = metis.part_graph(g_nx, parts)
    return {n: part_vec[i] for i, n in enumerate(g_nx.nodes())}


def build_cluster_patches(data, mapping: Dict[int, int]) -> List[Dict[str, Any]]:
    edge_index = data.edge_index
    clusters = {}
    for v, cid in mapping.items():
        clusters.setdefault(cid, set()).add(v)

    patches = []
    for cid, core in clusters.items():
        core = set(core)
        # add 1‑hop boundary overlap
        mask_core = torch.isin(edge_index[0], torch.tensor(sorted(core)))
        boundary = set(edge_index[1, mask_core].tolist())
        patch_nodes = sorted(core | boundary)
        patches.append({"nodes": torch.tensor(patch_nodes, dtype=torch.long), "center": None})
    return patches