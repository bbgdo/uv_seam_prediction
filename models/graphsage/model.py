import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class UVSeamGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 6,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = SAGEConv(node_in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # edge representation: [h_i || h_j || edge_attr] → logit
        edge_repr_dim = hidden_dim * 2 + edge_in_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # [N, node_in_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,   # [E, edge_in_dim]
        chunk_size: int = 100_000,
    ) -> torch.Tensor:             # [E] logits
        h = self.act(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = self.act(self.conv2(h, edge_index))

        src, dst = edge_index[0], edge_index[1]

        # chunked MLP to avoid OOM on large meshes — materialising [E, 260] all at once
        # can exceed VRAM for high-poly models
        chunks = []
        for i in range(0, src.shape[0], chunk_size):
            s = src[i:i + chunk_size]
            d = dst[i:i + chunk_size]
            ea = edge_attr[i:i + chunk_size]
            chunks.append(self.edge_mlp(torch.cat([h[s], h[d], ea], dim=-1)).squeeze(-1))

        return torch.cat(chunks, dim=0)
