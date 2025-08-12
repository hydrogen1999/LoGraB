from typing import Iterable, Dict, Any
import numpy as np
import torch
from .d_hop import get_d_hop_patch

def random_seed_patches(num_nodes: int, seeds: int, d: int, edge_index: torch.Tensor) -> Iterable[Dict[str, Any]]:
    chosen = np.random.choice(num_nodes, seeds, replace=False)
    for v in chosen:
        yield get_d_hop_patch(int(v), d, edge_index, num_nodes)