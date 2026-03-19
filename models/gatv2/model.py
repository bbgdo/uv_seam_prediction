import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class DualGATv2(nn.Module):
    """GATv2 for edge classification via dual graph node classification.

    On the dual graph, each node = original edge, each edge = face adjacency.
    Node features = 11-dim artistic edge features.
    Output = per-node (= per-original-edge) seam probability logit.
    """

    def __init__(
        self,
        in_dim: int = 11,
        hidden_dim: int = 64,
        heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # first layer: in_dim -> hidden_dim * heads
        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # middle layers: hidden_dim * heads -> hidden_dim * heads (with residuals)
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # final conv: single head for output
        self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # residual for layers where dims match (middle layers)
            if i > 0 and h.shape == x.shape:
                h = h + x
            x = h

        return self.classifier(x).squeeze(-1)
