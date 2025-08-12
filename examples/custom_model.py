# Usage:
#   python -m lograb eval --task nodeclf --instance ... #     --model lograb.examples.custom_model:build --model-cfg my_model.yml
from __future__ import annotations
import torch.nn as nn
from torch_geometric.nn import SAGEConv

def build(in_dim: int, out_dim: int, hid: int = 128, layers: int = 3, dropout: float = 0.3, **kw):
    class StackedSAGE(nn.Module):
        def __init__(self, in_dim, out_dim, hid, layers, dropout):
            super().__init__()
            self.layers = nn.ModuleList()
            dims = [in_dim] + [hid]*(layers-1) + [out_dim]
            for i in range(len(dims)-1):
                self.layers.append(SAGEConv(dims[i], dims[i+1]))
            self.dp = nn.Dropout(dropout)
        def forward(self, x, edge_index):
            for i, conv in enumerate(self.layers):
                x = conv(x, edge_index)
                if i < len(self.layers)-1:
                    x = x.relu()
                    x = self.dp(x)
            return x
    return StackedSAGE(in_dim, out_dim, hid, layers, dropout)