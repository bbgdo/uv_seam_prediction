from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshConv(nn.Module):
    """MeshCNN-style 4-neighbor edge convolution with symmetric face handling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(5 * in_channels, out_channels)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        edge_count, channels = x.shape
        padded = torch.cat([x, x.new_zeros(1, channels)], dim=0)
        safe_neighbors = neighbors.clamp(min=0)
        nb = padded[safe_neighbors]
        nb = nb.masked_fill((neighbors < 0).unsqueeze(-1), 0.0)

        face_a = nb[:, 0:2, :]
        face_b = nb[:, 2:4, :]

        desc_a = torch.cat([
            face_a[:, 0, :] + face_a[:, 1, :],
            torch.abs(face_a[:, 0, :] - face_a[:, 1, :]),
        ], dim=1)
        desc_b = torch.cat([
            face_b[:, 0, :] + face_b[:, 1, :],
            torch.abs(face_b[:, 0, :] - face_b[:, 1, :]),
        ], dim=1)

        face_sum = desc_a + desc_b
        face_diff = torch.abs(desc_a - desc_b)
        combined = torch.cat([x, face_sum, face_diff], dim=1)
        return self.proj(combined.reshape(edge_count, 5 * channels))


class MeshConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv = MeshConv(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, neighbors)
        h = self.norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return x + h if x.shape == h.shape else h


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.fuse = nn.Linear(in_channels, out_channels)
        self.conv = MeshConvBlock(out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, skip], dim=1)
        h = F.relu(self.fuse(h))
        return self.conv(h, neighbors)
