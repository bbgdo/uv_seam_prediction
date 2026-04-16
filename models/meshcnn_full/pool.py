from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.meshcnn_full.mesh import CollapseHistory, MutableMeshTopology


class MeshPool(nn.Module):
    """Learned edge-collapse pooling over a mutable triangle topology."""

    def __init__(
        self,
        channels: int,
        target_ratio: float = 0.85,
        min_edges: int = 32,
        max_collapses: int | None = 2048,
    ):
        super().__init__()
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError('target_ratio must be in (0, 1]')
        self.target_ratio = float(target_ratio)
        self.min_edges = int(min_edges)
        self.max_collapses = max_collapses
        self.scorer = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        topology: MutableMeshTopology,
    ) -> tuple[torch.Tensor, MutableMeshTopology, CollapseHistory]:
        device = x.device
        pooled_topology = topology.clone()
        old_edges = pooled_topology.unique_edges.copy()
        old_count = int(len(old_edges))
        target_edges = max(self.min_edges, int(round(old_count * self.target_ratio)))
        target_edges = min(target_edges, old_count)

        if target_edges >= old_count:
            history = CollapseHistory(
                old_edges=torch.as_tensor(old_edges, dtype=torch.long, device=device),
                new_edges=torch.as_tensor(old_edges, dtype=torch.long, device=device),
                old_to_new=torch.arange(old_count, dtype=torch.long, device=device),
                collapsed_edges=[],
                old_edge_count=old_count,
                new_edge_count=old_count,
            )
            return x, pooled_topology, history

        score_logits = self.scorer(x).squeeze(-1)
        scores = score_logits.detach().cpu().numpy()
        candidate_order = np.argsort(scores)
        candidate_keys = [
            (int(old_edges[idx, 0]), int(old_edges[idx, 1]))
            for idx in candidate_order
        ]

        old_to_current = np.arange(old_count, dtype=np.int64)
        collapsed: list[tuple[int, int]] = []
        collapse_budget = self.max_collapses if self.max_collapses is not None else old_count

        for edge_key in candidate_keys:
            if pooled_topology.edge_count <= target_edges or len(collapsed) >= collapse_budget:
                break
            edge_idx = pooled_topology.edge_key_to_idx.get(edge_key)
            if edge_idx is None or not pooled_topology.is_valid_collapse(edge_idx):
                continue
            record = pooled_topology.collapse_edge(edge_idx)
            valid = old_to_current >= 0
            remapped = record.old_to_new[old_to_current[valid]]
            old_to_current[valid] = remapped
            collapsed.append(record.edge_key)

        new_count = pooled_topology.edge_count
        old_to_new = torch.as_tensor(old_to_current, dtype=torch.long, device=device)
        x_new = x.new_zeros((new_count, x.shape[1]))
        counts = x.new_zeros((new_count, 1))
        valid = old_to_new >= 0
        if bool(valid.any()):
            dst = old_to_new[valid]
            weights = torch.sigmoid(score_logits[valid]).unsqueeze(1).clamp_min(1e-4)
            x_new.index_add_(0, dst, x[valid] * weights)
            counts.index_add_(0, dst, weights)
            x_new = x_new / counts.clamp_min(1e-4)

        history = CollapseHistory(
            old_edges=torch.as_tensor(old_edges, dtype=torch.long, device=device),
            new_edges=torch.as_tensor(pooled_topology.unique_edges, dtype=torch.long, device=device),
            old_to_new=old_to_new,
            collapsed_edges=collapsed,
            old_edge_count=old_count,
            new_edge_count=new_count,
        )
        return x_new, pooled_topology, history
