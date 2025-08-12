from typing import Dict, Any, Optional, Literal
import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh

LaplacianType = Literal["unnormalized", "normalized"]

def _build_sparse_adj(edge_index_np: np.ndarray, num_nodes: int) -> csr_matrix:
    row, col = edge_index_np
    data = np.ones(row.shape[0], dtype=np.float64)
    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A

def _unnormalized_laplacian(A: csr_matrix) -> csr_matrix:
    d = np.ravel(A.sum(axis=1))
    D = diags(d)
    return D - A

def _normalized_laplacian(A: csr_matrix) -> csr_matrix:
    d = np.ravel(A.sum(axis=1))
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = diags(inv_sqrt)
    I = identity(A.shape[0], format="csr")
    return I - (D_inv_sqrt @ A @ D_inv_sqrt)

def spectral_embed(edge_index: torch.Tensor, num_nodes: int, k: int, sigma: float,
                   laplacian: LaplacianType = "unnormalized") -> Optional[Dict[str, Any]]:
    if num_nodes <= 1:
        return None
    k_eff = max(1, min(k, num_nodes - 1))

    ei_np = edge_index.detach().cpu().numpy()
    A = _build_sparse_adj(ei_np, num_nodes)
    if laplacian == "normalized":
        L = _normalized_laplacian(A); which = "SM"
    else:
        L = _unnormalized_laplacian(A); which = "SM"

    want = min(k_eff + 1, max(1, num_nodes - 1))
    try:
        evals, evecs = eigsh(L, k=want, which=which)
    except Exception:
        evals, evecs = eigsh(L + 1e-3 * identity(num_nodes, format="csr"),
                             k=want, which="LM", sigma=0.0)

    evecs = evecs[:, :k_eff].astype(np.float32)
    if sigma > 0:
        evecs = evecs + np.random.normal(scale=sigma, size=evecs.shape).astype(np.float32)

    evals = evals.astype(np.float32)
    return {"eigvec": torch.from_numpy(evecs), "eigval": torch.from_numpy(evals[:k_eff + 1])}
