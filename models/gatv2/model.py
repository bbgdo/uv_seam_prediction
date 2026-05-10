import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class DualGATv2(nn.Module):

    def __init__(
        self,
        in_dim: int = 18,
        hidden_dim: int = 32,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be at least 1')
        if hidden_dim < 2:
            raise ValueError('hidden_dim must be at least 2')
        if heads < 1:
            raise ValueError('heads must be at least 1')

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        layer_in_dim = in_dim
        for layer_idx in range(num_layers):
            is_last_attention = layer_idx == num_layers - 1
            layer_heads = 1 if is_last_attention else heads
            concat = not is_last_attention
            layer_out_dim = hidden_dim if is_last_attention else hidden_dim * heads

            self.convs.append(
                GATv2Conv(
                    layer_in_dim,
                    hidden_dim,
                    heads=layer_heads,
                    concat=concat,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(layer_out_dim))
            layer_in_dim = layer_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if i > 0 and h.shape == x.shape:
                h = h + x
            x = h

        return self.classifier(x).squeeze(-1)
