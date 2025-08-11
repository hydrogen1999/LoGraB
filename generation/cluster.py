from typing import Dict, List, Any
from collections import defaultdict
import torch
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph

# Try METIS; if unavailable, fall back gracefully
try:
    import metis  # type: ignore
    _HAS_METIS = True
except Exception:
    _HAS_METIS = False


def partition_metis(data, parts: int = 64) -> Dict[int, int]:
    """
    Return mapping node -> cluster id.
    Uses METIS if available; otherwise falls back to greedy modularity communities.
    """
    g_nx = to_networkx(data, to_undirected=True)
    if _HAS_METIS:
        _, part_vec = metis.part_graph(g_nx, parts)
        return {n: int(part_vec[i]) for i, n in enumerate(g_nx.nodes())}

    # Fallback: greedy modularity communities, then fold into `parts` bins
    comms = list(nx.algorithms.community.greedy_modularity_communities(g_nx, weight=None))
    mapping: Dict[int, int] = {}
    cid = 0
    for C in comms:
        for n in C:
            mapping[int(n)] = cid % parts
        cid += 1
    return mapping


def build_cluster_patches(data, mapping: Dict[int, int]) -> List[Dict[str, Any]]:
    """
    Build patches by taking each cluster core and expanding with a 1-hop boundary.
    Returns nodes as a torch.long tensor (global ids); edges will be built later via `subgraph`.
    """
    clusters: Dict[int, set] = defaultdict(set)
    for v, cid in mapping.items():
        clusters[int(cid)].add(int(v))

    edge_index = data.edge_index
    patches: List[Dict[str, Any]] = []

    for cid, core in clusters.items():
        core_t = torch.tensor(sorted(core), dtype=torch.long)
        # 1-hop neighbors around core (either endpoint in core)
        mask = torch.isin(edge_index[0], core_t) | torch.isin(edge_index[1], core_t)
        boundary_nodes = torch.unique(edge_index[:, mask])
        patch_nodes = torch.unique(torch.cat([core_t, boundary_nodes]))
        patches.append({"nodes": patch_nodes, "center": None})
    return patches
