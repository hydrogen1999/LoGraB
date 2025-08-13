from __future__ import annotations
from pathlib import Path
import gzip, json
from typing import Dict, List, Tuple, Set
import numpy as np
import torch
from scipy.linalg import svd
from .utils import load_metadata

def _load_rows(instance: Path) -> List[dict]:
    rows = []
    with gzip.open(instance / "patches.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def _procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Tìm R trực chuẩn sao cho A R ≈ B (Orthogonal Procrustes)
    U, _, Vt = svd(A.T @ B, full_matrices=False)
    R = U @ Vt
    return R.astype(np.float32)

def _eigen_sync(R_ij: Dict[Tuple[int,int], np.ndarray],
                w_ij: Dict[Tuple[int,int], float],
                m: int, k: int) -> List[np.ndarray]:
    # M là ma trận khối (m*k) x (m*k)
    M = np.zeros((m*k, m*k), dtype=np.float32)
    for (i, j), Rij in R_ij.items():
        w = float(w_ij[(i, j)])
        M[i*k:(i+1)*k, j*k:(j+1)*k] = w * Rij
        M[j*k:(j+1)*k, i*k:(i+1)*k] = w * Rij.T
    # Lấy k trị riêng lớn nhất
    vals, vecs = np.linalg.eigh(M)
    idx = np.argsort(vals)[::-1][:k]
    V = vecs[:, idx]   # (m*k, k)
    R_list = []
    for i in range(m):
        block = V[i*k:(i+1)*k, :]  # k×k
        Qi, _ = np.linalg.qr(block)
        R_list.append(Qi.astype(np.float32))
    return R_list

def write_pred_eigsync(instance: Path, out_path: Path, k_recon: int = 10):
    meta = load_metadata(instance)
    rows = _load_rows(instance)

    # Tập nút thật trong instance
    V_true: Set[int] = set()
    for r in rows:
        V_true.update(int(x) for x in r["nodes_global"])
    V_true = sorted(V_true)
    gid2pos = {g: i for i, g in enumerate(V_true)}
    N = len(V_true)

    # Gom spec của mỗi patch
    k = None
    patches = []
    for r in rows:
        gnodes = [int(x) for x in r["nodes_global"]]
        if "eigvec" not in r or r["eigvec"] is None:
            # bỏ patch không có embedding phổ
            continue
        X = torch.tensor(r["eigvec"], dtype=torch.float32).numpy()  # n_i × k
        if k is None:
            k = X.shape[1]
        patches.append((gnodes, X))
    if k is None or k < 1 or not patches:
        raise RuntimeError("Không có spectral features trong patches.")

    # Ước lượng R_ij trên overlap ≥ 2
    R_ij, W = {}, {}
    for i in range(len(patches)):
        Vi, Xi = patches[i]
        map_i = {g: t for t, g in enumerate(Vi)}
        set_i = set(Vi)
        for j in range(i+1, len(patches)):
            Vj, Xj = patches[j]
            S = list(set_i & set(Vj))
            if len(S) < 2:
                continue
            map_j = {g: t for t, g in enumerate(Vj)}
            Ai = np.stack([Xi[map_i[g]] for g in S], axis=0)  # m×k
            Bj = np.stack([Xj[map_j[g]] for g in S], axis=0)
            Rij = _procrustes(Ai, Bj)
            R_ij[(i, j)] = Rij
            W[(i, j)] = float(len(S))

    if not R_ij:
        raise RuntimeError("Không có cặp patch overlap đủ để đồng bộ (eigsync).")

    # Eigen-synchronization → R_i cho từng patch
    R_list = _eigen_sync(R_ij, W, m=len(patches), k=k)

    # Nhúng toàn cục cho mỗi node bằng trung bình các biểu diễn đã đồng bộ
    Z = np.zeros((N, k), dtype=np.float32)
    C = np.zeros((N,), dtype=np.float32)
    for i, (Vi, Xi) in enumerate(patches):
        Ri = R_list[i]
        Xi_sync = Xi @ Ri  # n_i × k
        for t, g in enumerate(Vi):
            if g not in gid2pos:
                continue
            idx = gid2pos[g]
            Z[idx] += Xi_sync[t]
            C[idx] += 1.0
    C[C == 0] = 1.0
    Z = Z / C[:, None]

    # kNN theo cosine trong V_true để dựng cạnh dự đoán
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    Zc = Z / norms
    from heapq import nlargest
    E_pred = set()
    for u in range(N):
        s = (Zc @ Zc[u:u+1, :].T).ravel()
        cand = [(v, float(s[v])) for v in range(N) if v != u]
        top = nlargest(k_recon, cand, key=lambda x: x[1])
        for v, _ in top:
            a, b = (V_true[u], V_true[v]) if V_true[u] < V_true[v] else (V_true[v], V_true[u])
            E_pred.add((a, b))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for a, b in sorted(E_pred):
            f.write(f"{a} {b}\n")
    print(f"[eigsync] wrote {out_path} with {len(E_pred)} edges; k={k}, kNN={k_recon}")