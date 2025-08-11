from typing import Dict, Any, Optional
import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh


def _build_sparse_adj(edge_index_np: np.ndarray, num_nodes: int) -> csr_matrix:
    # edge_index: shape [2, E], undirected expected (we symmetrize anyway)
    row, col = edge_index_np
    data = np.ones(row.shape[0], dtype=np.float64)
    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)  # ensure symmetry
    A.setdiag(0)        # drop self-loops if any
    A.eliminate_zeros()
    return A


def _normalized_laplacian(A: csr_matrix) -> csr_matrix:
    d = np.ravel(A.sum(axis=1))
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = diags(inv_sqrt)
    # L = I - D^{-1/2} A D^{-1/2}
    I = identity(A.shape[0], format="csr")
    return I - (D_inv_sqrt @ A @ D_inv_sqrt)


def spectral_embed(edge_index: torch.Tensor, num_nodes: int, k: int, sigma: float) -> Optional[Dict[str, Any]]:
    """
    Return dict with k eigenvectors (+noise) and k+1 eigenvalues of the normalized Laplacian.
    Handles small patches safely and uses robust sparse solvers.
    """
    if num_nodes <= 1:
        return None

    k_eff = max(1, min(k, num_nodes - 1))  # at most n-1 non-zero eigenvalues for connected components

    # Build sparse normalized Laplacian
    ei_np = edge_index.detach().cpu().numpy()
    A = _build_sparse_adj(ei_np, num_nodes)
    L = _normalized_laplacian(A)

    # Compute (k_eff + 1) smallest magnitude eigenpairs for stability
    want = min(k_eff + 1, max(1, num_nodes - 1))
    try:
        evals, evecs = eigsh(L, k=want, which="SM")
    except Exception:
        # Shift + invert as fallback
        evals, evecs = eigsh(L + 1e-3 * identity(num_nodes, format="csr"),
                             k=want, which="LM", sigma=0.0)

    # Truncate to k_eff eigenvectors (drop the extra one used for residual/error diagnostics)
    evecs = evecs[:, :k_eff].astype(np.float32)
    if sigma > 0:
        evecs = evecs + np.random.normal(scale=sigma, size=evecs.shape).astype(np.float32)

    evals = evals.astype(np.float32)
    return {
        "eigvec": torch.from_numpy(evecs),
        "eigval": torch.from_numpy(evals[:k_eff + 1])  # keep k+1 evals if available
    }
