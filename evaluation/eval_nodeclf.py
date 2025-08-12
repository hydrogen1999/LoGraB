from __future__ import annotations
from pathlib import Path
import gzip, json, yaml, pathlib
from typing import Dict, List, Any

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from .models import get_builder
from .utils import load_metadata, load_source_graph
from ..utils import set_global_seed
from sklearn.metrics import f1_score
import numpy as np

def _load_patches(instance: Path) -> List[Dict[str, Any]]:
    rows = []
    with gzip.open(instance / "patches.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def _ensure_splits(instance: Path, seed: int = 42):
    sp = instance / "splits.yml"
    if sp.exists():
        return yaml.safe_load(sp.read_text())
    from ..splits import compute_patch_splits
    return compute_patch_splits(instance, seed=seed)

def _build_patch_graph(row: Dict[str, Any], data, add_spec: bool = True) -> Data:
    l2g = row.get("local_to_global_map", None)
    if l2g is None:
        local2global = {i: int(g) for i, g in enumerate(row["nodes_global"])}
    else:
        local2global = {int(k): int(v) for k, v in l2g.items()}

    globals_sorted = [local2global[i] for i in range(len(local2global))]
    node_idx = torch.tensor(globals_sorted, dtype=torch.long)

    sub_ei, _, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)
    x_orig = data.x[node_idx].to(torch.float)

    if add_spec and ("eigvec" in row) and (row["eigvec"] is not None):
        spec = torch.tensor(row["eigvec"], dtype=torch.float)
        if spec.dim() == 1:
            spec = spec.unsqueeze(1)
        x = torch.cat([x_orig, spec], dim=1)
    else:
        x = x_orig

    y = data.y[node_idx].long()
    d = Data(x=x, edge_index=sub_ei, y=y)
    d.gidx = node_idx
    return d

def _epoch(model, loader, opt, device, train=True):
    model.train(train)
    total_loss = 0.0
    ys, ps = [], []
    ce = nn.CrossEntropyLoss()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = ce(out, batch.y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += float(loss.item()) * batch.num_nodes
            ys.append(batch.y.detach().cpu().numpy())
            ps.append(out.detach().cpu().argmax(-1).numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    micro = f1_score(y, p, average="micro")
    macro = f1_score(y, p, average="macro")
    return total_loss / sum(len(a) for a in ys), micro, macro

def eval_nodeclf(instance: Path, pred: Path, **hparams):
    set_global_seed(int(hparams.get('seed', 42)))
    meta = load_metadata(instance)
    ds_name = meta["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_source_graph(ds_name)
    if getattr(data.y, 'ndim', 1) > 1 and data.y.size(-1) == 1:
        data.y = data.y.view(-1)
    num_classes = int(data.y.max().item() + 1)

    rows = _load_patches(instance)
    splits = _ensure_splits(instance, seed=42)

    id2row = {r["id"]: r for r in rows}
    def build_list(ids: List[str]) -> List[Data]:
        lst = []
        for pid in ids:
            row = id2row.get(pid, None)
            if row is None:
                continue
            d = _build_patch_graph(row, data, add_spec=True)
            lst.append(d)
        return lst

    train_list = build_list(splits["train_patches"])
    val_list   = build_list(splits["val_patches"])
    test_list  = build_list(splits["test_patches"])

    if not (train_list and val_list and test_list):
        raise RuntimeError("Empty split lists; check splits.yml / patches.jsonl.gz")

    in_dim = train_list[0].num_node_features
    model_name = hparams.get('model', 'sage')
    model_cfg = {}
    cfg_path = hparams.get('model_cfg', None)
    if cfg_path is not None:
        model_cfg = yaml.safe_load(open(cfg_path, 'r').read()) or {}
    builder = get_builder(model_name)
    model = builder(in_dim=in_dim, out_dim=num_classes, **model_cfg).to(device)

    lr = float(hparams.get('lr', 1e-2))
    wd = float(hparams.get('wd', 5e-4))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    bs = int(hparams.get('batch', 32))
    epochs = int(hparams.get('epochs', 100))
    train_loader = DataLoader(train_list, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_list, batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_list, batch_size=bs, shuffle=False)

    ckpt = hparams.get('checkpoint', None)
    if ckpt is not None and pathlib.Path(ckpt).exists():
        state = torch.load(ckpt, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    best_val, best_state = -1.0, None
    for epoch in range(1, epochs+1):
        tr_loss, tr_micro, tr_macro = _epoch(model, train_loader, opt, device, train=True)
        val_loss, val_micro, val_macro = _epoch(model, val_loader, opt, device, train=False)
        if val_micro > best_val:
            best_val = val_micro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_micro, te_macro = _epoch(model, test_loader, opt, device, train=False)
    print(f"[nodeclf] F1-micro {te_micro:.4f}  F1-macro {te_macro:.4f}")

    save_emb = hparams.get('save_embeddings', None)
    if save_emb is not None:
        model.eval()
        all_list = train_list + val_list + test_list
        with torch.no_grad():
            num_nodes = data.num_nodes
            tmp = all_list[0].to(device)
            tmp_out = model(tmp.x, tmp.edge_index)
            dim = tmp_out.size(-1)
            S = torch.zeros((num_nodes, dim), dtype=torch.float32)
            C = torch.zeros((num_nodes,), dtype=torch.float32)
            for dpatch in all_list:
                dpatch = dpatch.to(device)
                out = model(dpatch.x, dpatch.edge_index).detach().cpu()
                gidx = dpatch.gidx.detach().cpu().long()
                S[gidx] += out
                C[gidx] += 1.0
            C[C == 0] = 1.0
            E = S / C.unsqueeze(-1)
            from pathlib import Path as _P
            _P(save_emb).parent.mkdir(parents=True, exist_ok=True)
            torch.save(E, save_emb)
        print(f"[nodeclf] Saved embeddings to {save_emb}")