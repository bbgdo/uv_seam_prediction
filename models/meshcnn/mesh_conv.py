"""
MeshConv: edge convolution with fixed 4-neighbor structure.

Each edge in a manifold triangulated mesh has exactly 4 neighboring edges
(2 from each incident triangle). MeshConv exploits this fixed topology
to apply a learnable transformation over the symmetric neighborhood.

The key invariance trick: the 4 neighbors are split into 2 pairs (one pair
per incident triangle). Within each pair, features are sorted element-wise
so the result is invariant to the arbitrary labeling of pair members.

Reference: MeshCNN (Hanocka et al., arXiv:1809.05910).
"""

import torch
import torch.nn as nn


class MeshConv(nn.Module):
    """Convolution on mesh edges using the fixed 4-neighbor structure.

    Gathers features from each edge's 4 topological neighbors, applies
    element-wise sorting within each pair for order invariance, then
    projects the concatenation [self || pair1 || pair2] to the output dim.

    Args:
        in_channels:  feature dimension of input edges
        out_channels: feature dimension of output edges
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # input to FC: self (C) + pair1_sorted (2C) + pair2_sorted (2C) = 5C
        self.fc = nn.Linear(5 * in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,           # [E, in_channels]
        neighbors: torch.Tensor,   # [E, 4]  int64, -1 = missing (boundary)
    ) -> torch.Tensor:             # [E, out_channels]
        E, C = x.shape

        # safe gather: append a zero-feature row so -1 indices map to zeros
        x_padded = torch.cat([x, x.new_zeros(1, C)], dim=0)      # [E+1, C]
        safe_nb = neighbors.clamp(min=0)                           # [E, 4], -1 → 0 temporarily

        nb_feats = x_padded[safe_nb]                              # [E, 4, C]

        # zero out features for missing neighbors (boundary padding)
        missing = (neighbors < 0).unsqueeze(-1)                   # [E, 4, 1]
        nb_feats = nb_feats.masked_fill(missing, 0.0)

        # split into 2 pairs, one per incident triangle
        pair1 = nb_feats[:, 0:2, :]                               # [E, 2, C]
        pair2 = nb_feats[:, 2:4, :]                               # [E, 2, C]

        # sort within each pair element-wise → invariant to pair-member order
        pair1_sorted, _ = torch.sort(pair1, dim=1)                # [E, 2, C]
        pair2_sorted, _ = torch.sort(pair2, dim=1)                # [E, 2, C]

        combined = torch.cat([
            x,                              # [E, C]
            pair1_sorted.reshape(E, 2 * C), # [E, 2C]
            pair2_sorted.reshape(E, 2 * C), # [E, 2C]
        ], dim=1)                           # [E, 5C]

        return self.fc(combined)            # [E, out_channels]
