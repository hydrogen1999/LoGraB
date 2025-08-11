from typing import Dict, Any
import torch
from torch_geometric.utils import k_hop_subgraph


def get_d_hop_patch(node_idx: int, d: int, edge_index: torch.Tensor, num_nodes: int) -> Dict[str, Any]:
    """Return *d*-hop patch centred at *node_idx*."""
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(node_idx, d, edge_index,
                                                         relabel_nodes=True,
                                                         num_nodes=num_nodes)
    local2global = {int(l): int(g) for l, g in enumerate(subset)}
    return {
        "nodes": subset,
        "edge_index": sub_edge_index,
        "center": int(node_idx),
        "local_to_global_map": local2global,
    }