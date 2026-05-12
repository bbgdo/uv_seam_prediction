import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import sort_edge_index


GRAPH_SAGE_AGGREGATION = 'lstm'
SUPPORTED_GRAPH_SAGE_AGGREGATIONS = ('lstm', 'mean')


class DualGraphSAGE(nn.Module):

    def __init__(
        self,
        in_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        skip_connections: str = 'hidden',
        aggr: str = GRAPH_SAGE_AGGREGATION,
    ):
        super().__init__()
        if aggr not in SUPPORTED_GRAPH_SAGE_AGGREGATIONS:
            raise ValueError(f'aggr must be one of {SUPPORTED_GRAPH_SAGE_AGGREGATIONS}, got {aggr!r}')
        self.num_layers = num_layers
        self.dropout = dropout
        self.skip_connections = skip_connections

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggr))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.skips.append(nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity())

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.skips.append(nn.Identity())

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index):
        edge_index = sort_edge_index(edge_index, sort_by_row=False)

        for i, (conv, norm, skip) in enumerate(zip(self.convs, self.norms, self.skips)):
            residual = x
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.skip_connections == 'all':
                h = h + skip(residual)
            elif self.skip_connections == 'hidden' and i > 0 and h.shape == residual.shape:
                h = h + residual
            x = h

        return self.classifier(x).squeeze(-1)
