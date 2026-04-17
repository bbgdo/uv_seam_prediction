from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.meshcnn_full.mesh import CollapseHistory, MutableMeshTopology


def _topology_lineage(topology: MutableMeshTopology, edge_count: int) -> list[list[int]]:
    lineage = getattr(topology, '_meshcnn_orig_edge_ids', None)
    if lineage is None or len(lineage) != edge_count:
        return [[idx] for idx in range(edge_count)]
    out: list[list[int]] = []
    for item in lineage:
        if isinstance(item, (list, tuple, set, np.ndarray)):
            out.append([int(value) for value in item])
        else:
            out.append([int(item)])
    return out


def _remap_lineage(
    lineage: list[list[int]],
    old_to_new: np.ndarray,
    new_count: int,
) -> list[list[int]]:
    remapped: list[list[int]] = [[] for _ in range(new_count)]
    for old_idx, new_idx in enumerate(old_to_new):
        if int(new_idx) >= 0:
            remapped[int(new_idx)].extend(lineage[old_idx])
    return [sorted(set(values)) for values in remapped]


def _flatten_lineage(lineage: list[list[int]]) -> list[int]:
    return sorted({int(value) for values in lineage for value in values})


class MeshPool(nn.Module):
    """Learned edge-collapse pooling over a mutable triangle topology."""

    def __init__(
        self,
        channels: int,
        target_ratio: float = 0.85,
        min_edges: int = 32,
        max_collapses: int | None = 2048,
        max_attempts: int | None = None,
    ):
        super().__init__()
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError('target_ratio must be in (0, 1]')
        if max_collapses is not None and int(max_collapses) < 0:
            raise ValueError('max_collapses must be non-negative or None')
        if max_attempts is not None and int(max_attempts) < 0:
            raise ValueError('max_attempts must be non-negative or None')
        self.target_ratio = float(target_ratio)
        self.min_edges = int(min_edges)
        self.max_collapses = None if max_collapses is None else int(max_collapses)
        self.max_attempts = None if max_attempts is None else int(max_attempts)
        self.scorer = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )
        self.last_debug = None

    def get_last_debug(self):
        return self.last_debug

    def forward(
        self,
        x: torch.Tensor,
        topology: MutableMeshTopology,
    ) -> tuple[torch.Tensor, MutableMeshTopology, CollapseHistory]:
        debug = {
            'input_edges': 0,
            'output_edges': 0,
            'target_edges': 0,
            'candidate_count': 0,
            'attempted_collapses': 0,
            'max_attempts': 0,
            'successful_collapses': 0,
            'rejected_collapses': 0,
            'reject_boundary': 0,
            'reject_nonmanifold': 0,
            'reject_degenerate': 0,
            'stop_reason': None,
            'collapsed_norm_mean': None,
            'retained_norm_mean': None,
            'survivor_orig_edge_ids': None,
        }
        device = x.device
        pooled_topology = topology.clone()
        old_edges = pooled_topology.unique_edges.copy()
        old_count = int(len(old_edges))
        input_lineage = _topology_lineage(topology, old_count)
        debug['input_edges'] = old_count
        target_edges = max(self.min_edges, int(round(old_count * self.target_ratio)))
        target_edges = min(target_edges, old_count)
        debug['target_edges'] = target_edges

        if target_edges >= old_count:
            debug['output_edges'] = old_count
            debug['stop_reason'] = 'already_at_or_below_target'
            pooled_topology._meshcnn_orig_edge_ids = [list(values) for values in input_lineage]
            debug['survivor_orig_edge_ids'] = _flatten_lineage(input_lineage)
            self.last_debug = debug
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
        debug['candidate_count'] = len(candidate_keys)
        old_edge_key_to_idx = {
            (int(edge[0]), int(edge[1])): idx
            for idx, edge in enumerate(old_edges)
        }

        old_to_current = np.arange(old_count, dtype=np.int64)
        collapsed: list[tuple[int, int]] = []
        collapsed_old_indices: list[int] = []
        collapse_budget = self.max_collapses if self.max_collapses is not None else old_count
        max_attempts = self.max_attempts if self.max_attempts is not None else len(candidate_keys)
        max_attempts = min(max_attempts, len(candidate_keys))
        debug['max_attempts'] = max_attempts

        candidate_cursor = 0
        while pooled_topology.edge_count > target_edges:
            if len(collapsed) >= collapse_budget:
                debug['stop_reason'] = 'collapse_budget_reached'
                break
            if candidate_cursor >= len(candidate_keys):
                debug['stop_reason'] = 'candidate_exhausted'
                break
            if debug['attempted_collapses'] >= max_attempts:
                debug['stop_reason'] = 'max_attempts_reached'
                break

            edge_key = candidate_keys[candidate_cursor]
            candidate_cursor += 1
            debug['attempted_collapses'] += 1
            edge_idx = pooled_topology.edge_key_to_idx.get(edge_key)
            if edge_idx is None:
                debug['rejected_collapses'] += 1
                debug['reject_degenerate'] += 1
                continue
            collapse_error = pooled_topology.collapse_error(edge_idx)
            if collapse_error is not None:
                debug['rejected_collapses'] += 1
                error_text = collapse_error.lower()
                if 'boundary' in error_text:
                    debug['reject_boundary'] += 1
                elif 'non-manifold' in error_text or 'incident faces' in error_text:
                    debug['reject_nonmanifold'] += 1
                else:
                    debug['reject_degenerate'] += 1
                continue
            before_edges = pooled_topology.edge_count
            record = pooled_topology.collapse_edge(edge_idx)
            if pooled_topology.edge_count >= before_edges:
                raise RuntimeError(
                    f'edge collapse for {edge_key} did not reduce edge count '
                    f'({before_edges} -> {pooled_topology.edge_count})'
                )
            valid = old_to_current >= 0
            remapped = record.old_to_new[old_to_current[valid]]
            old_to_current[valid] = remapped
            collapsed.append(record.edge_key)
            collapsed_old_indices.append(old_edge_key_to_idx[edge_key])
            debug['successful_collapses'] += 1

        if debug['stop_reason'] is None:
            debug['stop_reason'] = 'target_reached'
        elif (
            debug['stop_reason'] == 'candidate_exhausted'
            and pooled_topology.edge_count > target_edges
            and debug['successful_collapses'] == 0
        ):
            debug['stop_reason'] = 'stagnated_no_valid_collapses'

        new_count = pooled_topology.edge_count
        old_to_new = torch.as_tensor(old_to_current, dtype=torch.long, device=device)
        if collapsed_old_indices:
            collapsed_idx = torch.as_tensor(collapsed_old_indices, dtype=torch.long, device=device)
            debug['collapsed_norm_mean'] = float(score_logits[collapsed_idx].detach().mean().item())
        retained_mask = old_to_new >= 0
        if bool(retained_mask.any()):
            debug['retained_norm_mean'] = float(score_logits[retained_mask].detach().mean().item())
        output_lineage = _remap_lineage(input_lineage, old_to_current, new_count)
        pooled_topology._meshcnn_orig_edge_ids = output_lineage
        debug['survivor_orig_edge_ids'] = _flatten_lineage(output_lineage)
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
        debug['output_edges'] = new_count
        self.last_debug = debug
        return x_new, pooled_topology, history
