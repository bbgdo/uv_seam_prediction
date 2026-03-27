import torch
import torch.nn as nn
import torch.nn.functional as F

from models.meshcnn.mesh_conv import MeshConv


class MeshCNNClassifier(nn.Module):
    """Stacked MeshConv layers for per-edge binary classification.

    No edge-collapse pooling — classifies at original mesh resolution.
    MeshConv expands to 5*in_channels internally, so hidden=64 implies
    a 320-dim intermediate — comparable to hidden=128 SAGEConv.
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
