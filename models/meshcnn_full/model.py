from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.meshcnn_full.layers import DecoderBlock, MeshConvBlock
from models.meshcnn_full.mesh import MeshCNNSample, MutableMeshTopology
from models.meshcnn_full.pool import MeshPool
from models.meshcnn_full.unpool import MeshUnpool


class MeshCNNSegmenter(nn.Module):
    """MeshCNN-style encoder-decoder for per-original-edge binary logits."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.2,
        pool_ratios: tuple[float, float] = (0.85, 0.75),
        min_edges: int = 32,
        max_pool_collapses: int | None = 2048,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.pool_ratios = tuple(float(r) for r in pool_ratios)

        self.stem = nn.Sequential(
            nn.Linear(self.in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )
        self.enc0 = MeshConvBlock(hidden_channels, dropout=dropout)
        self.pool0 = MeshPool(
            hidden_channels,
            target_ratio=self.pool_ratios[0],
            min_edges=min_edges,
            max_collapses=max_pool_collapses,
        )
        self.enc1 = MeshConvBlock(hidden_channels, dropout=dropout)
        self.pool1 = MeshPool(
            hidden_channels,
            target_ratio=self.pool_ratios[1],
            min_edges=min_edges,
            max_collapses=max_pool_collapses,
        )
        self.bottleneck = MeshConvBlock(hidden_channels, dropout=dropout)

        self.unpool = MeshUnpool()
        self.dec1 = DecoderBlock(2 * hidden_channels, hidden_channels, dropout=dropout)
        self.dec0 = DecoderBlock(2 * hidden_channels, hidden_channels, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, sample: MeshCNNSample) -> torch.Tensor:
        device = next(self.parameters()).device
        x = sample.edge_features.to(device)
        topology0 = MutableMeshTopology.from_sample(sample)

        h0 = self.stem(x)
        h0 = self.enc0(h0, topology0.neighbors_tensor(device))
        skip0 = h0

        p0, topology1, history0 = self.pool0(h0, topology0)
        h1 = self.enc1(p0, topology1.neighbors_tensor(device))
        skip1 = h1

        p1, topology2, history1 = self.pool1(h1, topology1)
        h2 = self.bottleneck(p1, topology2.neighbors_tensor(device))

        up1 = self.unpool(h2, history1)
        h = self.dec1(up1, skip1, topology1.neighbors_tensor(device))

        up0 = self.unpool(h, history0)
        h = self.dec0(up0, skip0, topology0.neighbors_tensor(device))

        return self.head(h).squeeze(-1)


def build_model_from_checkpoint_payload(payload: dict, device: torch.device | str) -> MeshCNNSegmenter:
    config = dict(payload.get('model_config', {}))
    if 'in_channels' not in config:
        metadata = payload.get('feature_metadata', {})
        feature_names = metadata.get('feature_names') or []
        if not feature_names:
            raise ValueError('checkpoint is missing in_channels and feature_names metadata')
        config['in_channels'] = len(feature_names)
    model = MeshCNNSegmenter(**config).to(device)
    state = payload.get('model_state', payload)
    model.load_state_dict(state)
    model.eval()
    return model
