# evaluation/seal_linkpred.py
from __future__ import annotations
from pathlib import Path
import gzip, json, random
from typing import List, Set, Tuple
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import GCNConv, global_max_pool, global_add_pool
from sklearn.metrics import roc_auc_score, average_precision_score

from .utils import load_metadata, load_source_graph


# -------------------------
# Fast co-occurrence (bit-pack)
# -------------------------
def _build_cooc_ei(rows: List[dict], num_nodes: int) -> torch.Tensor:
    """
    Xây H bằng cách nối mọi cặp (u,v) cùng xuất hiện trong *bất kỳ* patch nào.
    Dùng bit-pack (u<<32)|v để giảm overhead Python set.
    """
    E64: Set[int] = set()
    for r in rows:
        nodes = sorted(set(int(x) for x in r["nodes_global"]))
        # itertools.combinations ở C rất nhanh; không cần hai vòng for thuần Python
        from itertools import combinations
        for u, v in combinations(nodes, 2):
            E64.add((u << 32) | v)
    if not E64:
        raise RuntimeError("Co-occurrence graph empty.")
    src = [e >> 32 for e in E64]
    dst = [e & 0xffffffff for e in E64]
    # undirected
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return ei


def _load_rows(instance: Path) -> List[dict]:
    rows = []
    with gzip.open(instance / "patches.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# -------------------------
# DRNL (double-radius node labeling)
# -------------------------
def _drnl(edge_index: torch.Tensor, u: int, v: int, n: int) -> torch.Tensor:
    from collections import deque, defaultdict
    INF = 10**9
    adj = defaultdict(list)
    for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[a].append(b)

    def bfs(s):
        dist = [INF] * n
        q = deque([s]); dist[s] = 0
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[y] == INF:
                    dist[y] = dist[x] + 1
                    q.append(y)
        return dist

    du, dv = bfs(u), bfs(v)
    labels = []
    for i in range(n):
        if du[i] == INF and dv[i] == INF:
            labels.append(0)
        else:
            m = min(du[i], dv[i])
            s = du[i] + dv[i]
            labels.append(1 + m + (s * (s + 1)) // 2)
    return torch.tensor(labels, dtype=torch.long)


# -------------------------
# Tạo subgraph cho 1 cặp (u,v)
# -------------------------
def _pair_subgraph(H_ei: torch.Tensor, u: int, v: int, x: torch.Tensor, hops: int = 2) -> Data:
    subset, sub_ei, mapping, _ = k_hop_subgraph([u, v], hops, H_ei, relabel_nodes=True)
    lu, lv = int(mapping[0]), int(mapping[1])  # chỉ số local của u,v
    labels = _drnl(sub_ei, lu, lv, subset.numel())

    d = Data(
        x=x[subset],
        edge_index=sub_ei,
        drnl=labels,
        # Lưu vị trí local 2 node mục tiêu, để pooling “target-aware”
        target_local=torch.tensor([lu, lv], dtype=torch.long),
    )
    return d


# -------------------------
# Mô hình: Target-Aware Pooling (đúng chuẩn SEAL)
# Lấy biểu diễn tại đúng 2 node (u, v) + summary subgraph
# -------------------------
class SEALTargetAware(nn.Module):
    def __init__(self, in_dim: int, hid: int, num_labels: int, readout: str = "max"):
        super().__init__()
        self.emb = nn.Embedding(num_labels, 32)
        self.gcn1 = GCNConv(in_dim + 32, hid)
        self.gcn2 = GCNConv(hid, hid)
        self.readout = readout
        # Đặc trưng target-aware: h_u, h_v, |h_u - h_v|, h_u * h_v, h_pool
        feat_dim = hid * 5
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 2),
        )

    def forward(self, x, edge_index, drnl, batch, target_local, ptr=None):
        z = torch.cat([x, self.emb(drnl)], dim=1)
        z = self.gcn1(z, edge_index).relu()
        z = self.gcn2(z, edge_index).relu()

        # Global summary của subgraph (theo graph)
        if self.readout == "add":
            h_pool = global_add_pool(z, batch)
        else:
            h_pool = global_max_pool(z, batch)  # mặc định

        # --- Target-aware gather: lấy đúng embedding tại node u,v trong từng subgraph ---
        # ptr: [num_graphs+1], offset node theo graph. (có trong PyG >= 2.x)
        if ptr is None:
            # Fallback nếu batch.ptr vắng: tính từ 'batch'
            # (chậm hơn; chỉ xảy ra khi PyG cũ)
            num_graphs = int(batch.max().item()) + 1
            counts = torch.bincount(batch, minlength=num_graphs)
            ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=batch.device)
            ptr[1:] = torch.cumsum(counts, dim=0)

        num_graphs = ptr.numel() - 1
        uv = target_local.view(num_graphs, 2)                     # [G,2]
        offs = ptr[:-1].unsqueeze(1).expand_as(uv)                # [G,2]
        idx = (uv + offs).reshape(-1)                             # [2G]
        z_uv = z.index_select(0, idx)                             # [2G, hid]
        h_u = z_uv[0::2]                                          # [G, hid]
        h_v = z_uv[1::2]                                          # [G, hid]

        feats = torch.cat([h_u, h_v, (h_u - h_v).abs(), h_u * h_v, h_pool], dim=1)
        logits = self.mlp(feats)
        return logits


# -------------------------
# API chạy
# -------------------------
def eval_linkpred_seal(
    instance: Path,
    hops: int = 2,
    epochs: int = 20,
    bs: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    readout: str = "max",
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    meta = load_metadata(instance)
    data = load_source_graph(meta["dataset"])
    rows = _load_rows(instance)

    num_nodes = data.num_nodes
    x = data.x.float() if getattr(data, "x", None) is not None else torch.zeros((num_nodes, 1))

    # H: co-occurrence graph (nhanh)
    H_ei = _build_cooc_ei(rows, num_nodes)

    # G (ground-truth)
    E_true: Set[Tuple[int, int]] = set()
    for a, b in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()):
        if a == b: continue
        E_true.add((min(a, b), max(a, b)))

    # Positives = cạnh thuộc G nhưng *chưa có* trong H (mới để dự đoán)
    H_edges = set()
    for a, b in zip(H_ei[0].tolist(), H_ei[1].tolist()):
        if a == b: continue
        H_edges.add((min(a, b), max(a, b)))
    pos = [e for e in E_true if e not in H_edges]

    # Negatives = cặp không phải cạnh của G
    def sample_negs(m):
        neg = set()
        while len(neg) < m:
            u = random.randrange(num_nodes); v = random.randrange(num_nodes)
            if u == v: continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in E_true: continue
            neg.add((a, b))
        return list(neg)

    npos = min(len(pos), 20000)
    random.shuffle(pos); pos = pos[:npos]
    neg = sample_negs(len(pos))

    pairs = pos + neg
    y = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int64)

    # Dataset subgraph
    ds: List[Data] = []
    for (u, v), yi in zip(pairs, y):
        d = _pair_subgraph(H_ei, u, v, x, hops=hops)
        d.y = torch.tensor(yi, dtype=torch.long)
        ds.append(d)

    # Train/Test split
    n = len(ds)
    idx = np.random.permutation(n)
    ntr = int(0.8 * n)
    tr_ds = [ds[i] for i in idx[:ntr]]
    te_ds = [ds[i] for i in idx[ntr:]]

    # Xác định num_labels theo DRNL lớn nhất của tập train (an toàn hơn hardcode)
    max_label = max(int(d.drnl.max()) for d in tr_ds) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SEALTargetAware(in_dim=x.size(-1), hid=64, num_labels=max_label, readout=readout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    pin = torch.cuda.is_available()
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    te_loader = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin)

    for _ in range(epochs):
        model.train()
        for batch in tr_loader:
            batch = batch.to(device)
            logits = model(
                batch.x, batch.edge_index, batch.drnl, batch.batch,
                batch.target_local, getattr(batch, "ptr", None)
            )
            loss = ce(logits, batch.y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(device)
            logits = model(
                batch.x, batch.edge_index, batch.drnl, batch.batch,
                batch.target_local, getattr(batch, "ptr", None)
            )
            prob1 = logits.softmax(dim=-1)[:, 1].detach().cpu().numpy().tolist()
            scores.extend(prob1)
            labels.extend(batch.y.view(-1).cpu().numpy().tolist())

    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    print(f"[seal-ta] AUROC {auroc:.4f}  AP {ap:.4f}  (pairs={len(labels)}, hops={hops}, epochs={epochs}, readout={readout})")