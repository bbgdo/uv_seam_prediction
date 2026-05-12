from __future__ import annotations

import torch
import torch.nn as nn


def _norm(channels: int, norm: str | None) -> nn.Module:
    if norm is None or norm == 'none':
        return nn.Identity()
    if norm == 'layer':
        return nn.LayerNorm(channels)
    raise ValueError(f'unsupported norm: {norm}')


class SparseMeshConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        norm: str | None = 'layer',
        residual: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.linear = nn.Linear(5 * self.in_channels, self.out_channels, bias=bias)
        self.norm = _norm(self.out_channels, norm)
        self.activation = nn.GELU()
        self.use_residual = bool(residual)
        if self.use_residual and self.in_channels != self.out_channels:
            self.residual_proj = nn.Linear(self.in_channels, self.out_channels, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, slot_mats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        s1, s2, s3, s4 = slot_mats
        a = torch.sparse.mm(s1, x)
        b = torch.sparse.mm(s2, x)
        c = torch.sparse.mm(s3, x)
        d = torch.sparse.mm(s4, x)
        patch = torch.cat(
            (
                x,
                torch.abs(a - c),
                a + c,
                torch.abs(b - d),
                b + d,
            ),
            dim=-1,
        )
        y = self.linear(patch)
        y = self.norm(y)
        if self.use_residual:
            y = y + self.residual_proj(x)
        return self.activation(y)


class SparseMeshConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv0 = SparseMeshConv(in_channels, out_channels)
        self.dropout0 = nn.Dropout(float(dropout))
        self.conv1 = SparseMeshConv(out_channels, out_channels)
        self.dropout1 = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, slot_mats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.dropout0(self.conv0(x, slot_mats))
        x = self.dropout1(self.conv1(x, slot_mats))
        return x


class SparseMeshPool(nn.Module):
    EPS = 1e-8

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(int(channels) // 2, 1)
        self.gate = nn.Sequential(
            nn.Linear(int(channels), hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self._last_debug: dict[str, int] = {}

    def forward(self, x: torch.Tensor, pool_map: torch.Tensor) -> torch.Tensor:
        pool_map = pool_map.coalesce()
        gates = torch.sigmoid(self.gate(x)).squeeze(-1)
        indices = pool_map.indices()
        values = pool_map.values() * gates.index_select(0, indices[1])
        gated_pool = torch.sparse_coo_tensor(
            indices,
            values,
            pool_map.shape,
            device=x.device,
            dtype=x.dtype,
        ).coalesce()
        mass = torch.sparse.mm(gated_pool, torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device))
        pooled = torch.sparse.mm(gated_pool, x) / mass.clamp_min(self.EPS)
        self._last_debug = {
            'input_edges': int(x.shape[0]),
            'output_edges': int(pooled.shape[0]),
            'successful_collapses': 0,
            'rejected_collapses': 0,
        }
        return pooled

    def get_last_debug(self) -> dict[str, int]:
        return dict(self._last_debug)


class SparseMeshUnpool(nn.Module):
    def forward(self, x: torch.Tensor, unpool_map: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(unpool_map, x)
