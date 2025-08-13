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
        L = _normalized_laplacian(A)
    else:
        L = _unnormalized_laplacian(A)

    # We want the smallest eigenpairs; fallback to shift-invert around 0 if needed
    want = min(k_eff + 1, max(1, num_nodes - 1))
    try:
        evals, evecs = eigsh(L, k=want, which="SM")
    except Exception:
        evals, evecs = eigsh(L, k=want, which="LM", sigma=0.0)

    # Sort ascending by eigenvalue, drop the trivial 0-eigenvector when present
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    start = 1 if evals[0] < 1e-9 else 0
    vecs = evecs[:, start:start + k_eff].astype(np.float32)
    vals = evals[start:start + k_eff + 1].astype(np.float32)

    if sigma > 0:
        vecs = (vecs + np.random.normal(scale=sigma, size=vecs.shape)).astype(np.float32)

    return {"eigvec": torch.from_numpy(vecs), "eigval": torch.from_numpy(vals)}