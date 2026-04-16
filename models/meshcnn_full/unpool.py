from __future__ import annotations

import torch
import torch.nn as nn

from models.meshcnn_full.mesh import CollapseHistory


class MeshUnpool(nn.Module):
    def forward(self, x: torch.Tensor, history: CollapseHistory) -> torch.Tensor:
        restored = x.new_zeros((history.old_edge_count, x.shape[1]))
        valid = history.old_to_new >= 0
        if bool(valid.any()):
            restored[valid] = x[history.old_to_new[valid]]
        return restored
