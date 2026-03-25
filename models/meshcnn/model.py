"""
MeshCNN edge classifier: stacked MeshConv layers → per-edge seam logit.

Simplified architecture without edge-collapse pooling — classifies edges at
original mesh resolution. The MeshConv operator provides the key inductive bias:
fixed 4-neighbor aggregation with topological structure. Pooling is omitted
because our task (seam prediction at full resolution) doesn't benefit from
hierarchical coarsening.

Reference: MeshCNN (Hanocka et al., arXiv:1809.05910).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.meshcnn.mesh_conv import MeshConv


class MeshCNNClassifier(nn.Module):
    """Stacked MeshConv layers for per-edge binary classification.

    Uses the same training conventions as DualGraphSAGE and DualGATv2:
    LayerNorm + activation + dropout after each layer, residuals from layer 2 onward,
    2-layer classifier MLP. Returns raw logits (use BCEWithLogitsLoss).

    Args:
        in_channels:     feature dimension of input edge features (default: 11)
        hidden_channels: internal feature dimension (default: 64)
        num_layers:      number of MeshConv layers (default: 4)
        dropout:         dropout probability (default: 0.3)

    Note on hidden_channels: MeshConv internally expands to 5 * in_channels before
    projecting to hidden_channels, so hidden=64 already implies a 320-dim intermediate
    representation in the first layer — comparable to hidden=128 SAGEConv.
    """

    def __init__(
        self,
        in_channels: int = 11,
        hidden_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(MeshConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(MeshConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,           # [E, in_channels]
        neighbors: torch.Tensor,   # [E, 4]  int64
    ) -> torch.Tensor:             # [E] logits
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, neighbors)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if i > 0 and h.shape == x.shape:
                h = h + x
            x = h

        return self.classifier(x).squeeze(-1)
