from __future__ import annotations

import torch
import torch.nn as nn

from models.meshcnn_full.mesh import MeshCNNSample
from models.meshcnn_full.sparse_layers import SparseMeshConvBlock, SparseMeshPool, SparseMeshUnpool
from models.meshcnn_full.sparse_precompute import get_or_build_sparse_cache, materialize_sparse_cache_for_step


def _channel_schedule(hidden_channels: int, levels: int) -> list[int]:
    multipliers = (1.0, 1.5, 2.0, 3.0, 4.0)
    channels = []
    for level in range(levels + 1):
        multiplier = multipliers[level] if level < len(multipliers) else multipliers[-1] + (level - len(multipliers) + 1)
        channels.append(max(1, int(round(hidden_channels * multiplier))))
    return channels


class SparseMeshUNetSegmenter(nn.Module):
    """Sparse static-topology U-Net for original-edge binary logits."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.2,
        pool_ratios: tuple[float, ...] = (0.85, 0.75),
        min_edges: int = 32,
        max_pool_collapses: int | None = 2048,
    ):
        super().__init__()
        del max_pool_collapses
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)
        self.pool_ratios = tuple(float(r) for r in pool_ratios)
        self.min_edges = int(min_edges)
        self.num_pools = len(self.pool_ratios)
        if self.num_pools < 1:
            raise ValueError('pool_ratios must contain at least one stage')

        channels = _channel_schedule(self.hidden_channels, self.num_pools)
        self.channels = tuple(channels)

        self.stem = nn.Sequential(
            nn.Linear(self.in_channels, channels[0]),
            nn.LayerNorm(channels[0]),
            nn.GELU(),
            nn.Linear(channels[0], channels[0]),
            nn.LayerNorm(channels[0]),
            nn.GELU(),
        )

        encoders = []
        pools = []
        for level in range(self.num_pools):
            enc_in = channels[level - 1] if level > 0 else channels[0]
            encoders.append(SparseMeshConvBlock(enc_in, channels[level], dropout=self.dropout))
            pools.append(SparseMeshPool(channels[level]))
        self.encoders = nn.ModuleList(encoders)
        self.pools = nn.ModuleList(pools)

        self.bottleneck = SparseMeshConvBlock(channels[self.num_pools - 1], channels[self.num_pools], dropout=self.dropout)
        self.unpool = SparseMeshUnpool()

        decoders = []
        current_channels = channels[self.num_pools]
        for level in reversed(range(self.num_pools)):
            decoders.append(SparseMeshConvBlock(current_channels + channels[level], channels[level], dropout=self.dropout))
            current_channels = channels[level]
        self.decoders = nn.ModuleList(decoders)

        head_hidden = max(channels[0] // 2, 1)
        self.head = nn.Sequential(
            nn.Linear(channels[0], head_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, sample: MeshCNNSample) -> torch.Tensor:
        device = next(self.parameters()).device
        x = sample.edge_features.to(device=device, dtype=torch.float32)
        cpu_cache = get_or_build_sparse_cache(
            sample,
            pool_ratios=self.pool_ratios,
            min_edges_per_level=self.min_edges,
        )
        step_cache = materialize_sparse_cache_for_step(cpu_cache, device)
        slot_levels = step_cache['slot_adj_levels']
        pool_maps = step_cache['pool_maps']
        unpool_maps = step_cache['unpool_maps']

        h = self.stem(x)
        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            h = encoder(h, slot_levels[level])
            skips.append(h)
            h = self.pools[level](h, pool_maps[level])

        h = self.bottleneck(h, slot_levels[self.num_pools])

        for decoder_idx, level in enumerate(reversed(range(self.num_pools))):
            h = self.unpool(h, unpool_maps[level])
            h = torch.cat((h, skips[level]), dim=-1)
            h = self.decoders[decoder_idx](h, slot_levels[level])

        return self.head(h).squeeze(-1)
