"""
Standalone UV seam inference worker — runs in an external Python process.

Usage:
    python run_inference.py <data.npz> <weights.pth> <threshold> <output.txt>

Reads mesh graph data from <data.npz>, runs the GraphSAGE edge classifier,
writes predicted seam edge indices (0-based into the unique-edge list) to
<output.txt>, one index per line.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


# Embedded model definition — mirrors models/graphsage/model.py exactly so this
# script is self-contained and works regardless of where the addon is installed.
class UVSeamGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int   = 6,
        edge_in_dim: int   = 11,
        hidden_dim:  int   = 128,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.conv1 = SAGEConv(node_in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim,  hidden_dim)
        self.conv3 = SAGEConv(hidden_dim,  hidden_dim)

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
        self.act     = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, chunk_size: int = 100_000):
        h = self.act(self.conv1(x, edge_index))
        h = self.dropout(h)
        h2 = self.act(self.conv2(h, edge_index))
        h2 = self.dropout(h2)
        h3 = self.act(self.conv3(h2, edge_index)) + h2  # residual
        h = self.dropout(h3)

        src, dst = edge_index[0], edge_index[1]
        chunks = []
        for i in range(0, src.shape[0], chunk_size):
            s  = src[i : i + chunk_size]
            d  = dst[i : i + chunk_size]
            ea = edge_attr[i : i + chunk_size]
            chunks.append(self.edge_mlp(torch.cat([h[s], h[d], ea], dim=-1)).squeeze(-1))

        return torch.cat(chunks, dim=0)


def main() -> None:
    if len(sys.argv) != 5:
        print("Usage: run_inference.py <data.npz> <weights.pth> <threshold> <output.txt>",
              file=sys.stderr)
        sys.exit(1)

    data_path, weights_path, threshold_str, output_path = sys.argv[1:]
    threshold = float(threshold_str)

    npz = np.load(data_path)
    x = torch.from_numpy(npz['x'])
    edge_index = torch.from_numpy(npz['edge_index'])
    edge_attr = torch.from_numpy(npz['edge_attr'])
    n_unique = int(npz['n_unique'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UVSeamGNN().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))

    seam_mask = torch.sigmoid(logits[:n_unique]) >= threshold
    seam_indices = seam_mask.nonzero(as_tuple=True)[0].tolist()

    with open(output_path, 'w') as f:
        f.write('\n'.join(map(str, seam_indices)))

    print(f"[UV Seam GNN] {len(seam_indices)} seam edges predicted out of {n_unique} total.")


if __name__ == '__main__':
    main()
