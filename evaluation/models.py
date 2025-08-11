from __future__ import annotations
from typing import Callable, Dict, Any
import importlib
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_model(name: str):
    def deco(fn: Callable[..., nn.Module]):
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Model '{name}' registered twice")
        _REGISTRY[key] = fn
        return fn
    return deco

def get_builder(name_or_path: str) -> Callable[..., nn.Module]:
    key = name_or_path.lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    # dynamic import path: "pkg.module:callable"
    if ":" not in name_or_path:
        raise KeyError(f"Unknown model '{name_or_path}'. Use a registered name or 'pkg.module:builder' path.")
    mod_path, call = name_or_path.split(":", 1)
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, call)
    if not callable(fn):
        raise TypeError(f"Imported object '{name_or_path}' is not callable")
    return fn

# ------- Built-in baselines -------

@register_model("gcn")
def _build_gcn(in_dim: int, out_dim: int, hid: int = 64, dropout: float = 0.5, **kw):
    class GCN(nn.Module):
        def __init__(self, in_dim, hid, out_dim, dropout):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hid)
            self.conv2 = GCNConv(hid, out_dim)
            self.dp = nn.Dropout(dropout)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.dp(x)
            x = self.conv2(x, edge_index)
            return x
    return GCN(in_dim, hid, out_dim, dropout)

@register_model("sage")
def _build_sage(in_dim: int, out_dim: int, hid: int = 64, dropout: float = 0.5, **kw):
    class SAGE(nn.Module):
        def __init__(self, in_dim, hid, out_dim, dropout):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hid)
            self.conv2 = SAGEConv(hid, out_dim)
            self.dp = nn.Dropout(dropout)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.dp(x)
            x = self.conv2(x, edge_index)
            return x
    return SAGE(in_dim, hid, out_dim, dropout)

@register_model("gat")
def _build_gat(in_dim: int, out_dim: int, hid: int = 32, heads: int = 4, dropout: float = 0.5, **kw):
    class GAT(nn.Module):
        def __init__(self, in_dim, hid, out_dim, heads, dropout):
            super().__init__()
            self.conv1 = GATConv(in_dim, hid, heads=heads, concat=True, dropout=dropout)
            self.conv2 = GATConv(hid*heads, out_dim, heads=1, concat=False, dropout=dropout)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x
    return GAT(in_dim, hid, out_dim, heads, dropout)

@register_model("gin")
def _build_gin(in_dim: int, out_dim: int, hid: int = 64, dropout: float = 0.5, **kw):
    import torch.nn as nn
    from torch_geometric.nn import GINConv
    class GIN(nn.Module):
        def __init__(self, in_dim, hid, out_dim, dropout):
            super().__init__()
            mlp1 = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid))
            mlp2 = nn.Sequential(nn.Linear(hid, out_dim))
            self.conv1 = GINConv(mlp1)
            self.conv2 = GINConv(mlp2)
            self.dp = nn.Dropout(dropout)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.dp(x)
            x = self.conv2(x, edge_index)
            return x
    return GIN(in_dim, hid, out_dim, dropout)