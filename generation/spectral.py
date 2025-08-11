from typing import Dict, Any
import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.linalg import eighs, splu


def spectral_embed(edge_index: torch.Tensor, num_nodes: int, k: int, sigma: float) -> Dict[str, Any]:
    """Return dict with *k* eigenvectors (+noise) and k+1 eigenvalues."""
    if num_nodes <= k:
        return None
    A = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes, dtype=np.float64)
    d = np.array(A.sum(axis=1)).flatten()
    L = np.diag(d) - A.toarray()
    try:
        evals, evecs = eighs(L, k=k + 1, which="SM")
    except Exception:
        # fallback: shift‑invert to guarantee convergence
        lu = splu(L + np.eye(num_nodes) * 1e-3)
        evals, evecs = eighs(L, k=k + 1, M=np.eye(num_nodes), sigma=0.0, OPinv=lu)
    evecs = evecs[:, :k].astype(np.float32)
    if sigma > 0:
        evecs += np.random.normal(scale=sigma, size=evecs.shape).astype(np.float32)
    return {"eigvec": torch.from_numpy(evecs), "eigval": torch.from_numpy(evals.astype(np.float32))}