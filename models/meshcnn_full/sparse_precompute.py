from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import torch

from models.meshcnn_full.mesh import MeshCNNSample


def _canonical_edge(a: int, b: int) -> tuple[int, int]:
    if a == b:
        raise ValueError(f'degenerate edge with repeated vertex id {a}')
    return (a, b) if a < b else (b, a)


def _empty_sparse(size: tuple[int, int], device: torch.device | str | None = None) -> torch.Tensor:
    indices = torch.empty((2, 0), dtype=torch.long, device=device)
    values = torch.empty((0,), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size, device=device).coalesce()


def _selector_from_pairs(rows: list[int], cols: list[int], size: int) -> torch.Tensor:
    if not rows:
        return _empty_sparse((size, size))
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()


def _binarize_sparse(matrix: torch.Tensor, *, zero_diagonal: bool = False) -> torch.Tensor:
    matrix = matrix.coalesce()
    if matrix._nnz() == 0:
        return matrix
    indices = matrix.indices()
    if zero_diagonal:
        keep = indices[0] != indices[1]
        indices = indices[:, keep]
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=indices.device)
    return torch.sparse_coo_tensor(indices, values, matrix.shape, device=matrix.device).coalesce()


def _row_normalize_sparse(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    matrix = matrix.coalesce()
    if matrix._nnz() == 0:
        return matrix
    indices = matrix.indices()
    values = matrix.values().to(dtype=torch.float32)
    row_mass = torch.zeros(matrix.shape[0], dtype=values.dtype, device=values.device)
    row_mass.scatter_add_(0, indices[0], values)
    values = values / row_mass.index_select(0, indices[0]).clamp_min(eps)
    return torch.sparse_coo_tensor(indices, values, matrix.shape, device=matrix.device).coalesce()


def build_slot_matrices(unique_edges: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build four MeshCNN-style sparse neighbor-slot selectors.

    Rows and columns are aligned with ``unique_edges``. Missing boundary slots are
    represented by zero rows in the corresponding sparse selector.
    """
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f'expected triangular faces with shape [F, 3], got {tuple(faces.shape)}')
    if unique_edges.ndim != 2 or unique_edges.shape[1] != 2:
        raise ValueError(f'expected unique_edges with shape [E, 2], got {tuple(unique_edges.shape)}')

    edges_cpu = unique_edges.detach().to(device='cpu', dtype=torch.long)
    faces_cpu = faces.detach().to(device='cpu', dtype=torch.long)
    edge_list = [(int(a), int(b)) for a, b in edges_cpu.tolist()]
    face_list = [[int(v) for v in face] for face in faces_cpu.tolist()]
    edge_lookup = {edge: idx for idx, edge in enumerate(edge_list)}
    edge_count = len(edge_list)

    incident: list[list[tuple[int, int]]] = [[] for _ in range(edge_count)]
    for face in face_list:
        u, v, w = face
        directed_edges = ((u, v, w), (v, w, u), (w, u, v))
        for a, b, third in directed_edges:
            key = _canonical_edge(a, b)
            edge_idx = edge_lookup.get(key)
            if edge_idx is None:
                raise KeyError(key)
            side = 0 if (a, b) == key else 1
            incident[edge_idx].append((side, third))

    slot_rows = [[], [], [], []]
    slot_cols = [[], [], [], []]
    for edge_idx, ((u, v), entries) in enumerate(zip(edge_list, incident)):
        if len(entries) > 2:
            raise ValueError(f'non-manifold edge {(u, v)}: {len(entries)} incident faces')

        left_values = [third for side, third in entries if side == 0]
        right_values = [third for side, third in entries if side == 1]
        left = left_values[0] if left_values else None
        right = right_values[0] if right_values else None
        if len(entries) == 2:
            if left is None:
                left = entries[0][1]
            if right is None:
                right = entries[1][1] if entries[1][1] != left else entries[0][1]

        if left is not None:
            slot_rows[0].append(edge_idx)
            slot_cols[0].append(edge_lookup[_canonical_edge(v, left)])
            slot_rows[1].append(edge_idx)
            slot_cols[1].append(edge_lookup[_canonical_edge(left, u)])
        if right is not None:
            slot_rows[2].append(edge_idx)
            slot_cols[2].append(edge_lookup[_canonical_edge(u, right)])
            slot_rows[3].append(edge_idx)
            slot_cols[3].append(edge_lookup[_canonical_edge(right, v)])

    return tuple(_selector_from_pairs(rows, cols, edge_count) for rows, cols in zip(slot_rows, slot_cols))  # type: ignore[return-value]


def build_line_adjacency(slot_mats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    edge_count = int(slot_mats[0].shape[0])
    if edge_count == 0:
        return _empty_sparse((0, 0), device=slot_mats[0].device)
    merged = sum((slot.coalesce() for slot in slot_mats), _empty_sparse((edge_count, edge_count), device=slot_mats[0].device))
    return _binarize_sparse(merged, zero_diagonal=True)


def _adjacency_sets(slot_mats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> list[set[int]]:
    edge_count = int(slot_mats[0].shape[0])
    adjacency = [set() for _ in range(edge_count)]
    for slot in slot_mats:
        slot = slot.coalesce()
        if slot._nnz() == 0:
            continue
        rows = slot.indices()[0].tolist()
        cols = slot.indices()[1].tolist()
        for row, col in zip(rows, cols):
            if row == col:
                continue
            adjacency[row].add(col)
            adjacency[col].add(row)
    return adjacency


def _match_groups(adjacency: list[set[int]], target_edges: int) -> list[list[int]]:
    groups = [[idx] for idx in range(len(adjacency))]
    current_adjacency = [set(neigh) for neigh in adjacency]
    target_edges = max(1, min(int(target_edges), len(groups)))

    while len(groups) > target_edges:
        reduction_needed = len(groups) - target_edges
        matched = [False] * len(groups)
        old_to_new = [-1] * len(groups)
        next_groups: list[list[int]] = []

        for idx in range(len(groups)):
            if matched[idx]:
                continue
            partner = -1
            if reduction_needed > 0:
                for candidate in sorted(current_adjacency[idx]):
                    if candidate != idx and not matched[candidate]:
                        partner = candidate
                        break
            if partner >= 0:
                matched[idx] = True
                matched[partner] = True
                old_to_new[idx] = len(next_groups)
                old_to_new[partner] = len(next_groups)
                next_groups.append(groups[idx] + groups[partner])
                reduction_needed -= 1
            else:
                matched[idx] = True
                old_to_new[idx] = len(next_groups)
                next_groups.append(groups[idx])

        if len(next_groups) == len(groups):
            break

        next_adjacency = [set() for _ in range(len(next_groups))]
        for row, neighbors in enumerate(current_adjacency):
            new_row = old_to_new[row]
            for col in neighbors:
                new_col = old_to_new[col]
                if new_row != new_col:
                    next_adjacency[new_row].add(new_col)
                    next_adjacency[new_col].add(new_row)
        groups = next_groups
        current_adjacency = next_adjacency

    return groups


def _pool_from_groups(groups: list[list[int]], edge_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[int] = []
    cols: list[int] = []
    assignment = torch.empty(edge_count, dtype=torch.long)
    for coarse_idx, group in enumerate(groups):
        for fine_idx in group:
            rows.append(coarse_idx)
            cols.append(fine_idx)
            assignment[fine_idx] = coarse_idx

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    pool = torch.sparse_coo_tensor(indices, values, (len(groups), edge_count)).coalesce()
    unpool = torch.sparse_coo_tensor(indices.flip(0), values, (edge_count, len(groups))).coalesce()
    unpool = _row_normalize_sparse(unpool)
    return pool, unpool, assignment


def _coarse_slots(
    slot_mats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    assignment: torch.Tensor,
    coarse_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = []
    for slot in slot_mats:
        slot = slot.coalesce()
        if slot._nnz() == 0:
            out.append(_empty_sparse((coarse_count, coarse_count)))
            continue
        indices = slot.indices()
        rows = assignment.index_select(0, indices[0])
        cols = assignment.index_select(0, indices[1])
        keep = rows != cols
        if not bool(torch.any(keep)):
            out.append(_empty_sparse((coarse_count, coarse_count)))
            continue
        coarse_indices = torch.stack((rows[keep], cols[keep]), dim=0)
        values = torch.ones(coarse_indices.shape[1], dtype=torch.float32)
        coarse = torch.sparse_coo_tensor(coarse_indices, values, (coarse_count, coarse_count)).coalesce()
        out.append(_row_normalize_sparse(_binarize_sparse(coarse, zero_diagonal=True)))
    return tuple(out)  # type: ignore[return-value]


def coarsen_slot_matrices(
    slot_mats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    target_edges: int,
    vertices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    del vertices
    edge_count = int(slot_mats[0].shape[0])
    if edge_count <= 1 or target_edges >= edge_count:
        groups = [[idx] for idx in range(edge_count)]
    else:
        groups = _match_groups(_adjacency_sets(slot_mats), target_edges)
    pool, unpool, assignment = _pool_from_groups(groups, edge_count)
    next_slots = _coarse_slots(slot_mats, assignment, int(pool.shape[0]))
    return pool, unpool, next_slots


def _iter_tensors(value: Any, path: str = 'cache') -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, f'{path}.{key}')
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from _iter_tensors(item, f'{path}[{idx}]')


def _drop_legacy_device_cache_entries(cache: dict[str, Any]) -> None:
    # Older runs stored CUDA sparse tensors under these keys on the sample-owned cache.
    cache.pop('_device_caches', None)
    cache.pop('device_caches', None)


def assert_sparse_cache_cpu_only(cpu_cache: dict[str, Any]) -> None:
    for forbidden_key in ('_device_caches', 'device_caches'):
        if forbidden_key in cpu_cache:
            raise AssertionError(f'persistent sparse cache must not contain {forbidden_key}')
    for path, tensor in _iter_tensors(cpu_cache):
        if tensor.device.type != 'cpu':
            raise AssertionError(f'persistent sparse cache tensor {path} is on {tensor.device}, expected cpu')


def build_sparse_cache(
    sample: MeshCNNSample,
    pool_ratios: tuple[float, ...] = (0.80, 0.64, 0.48),
    min_edges_per_level: int = 32,
) -> dict[str, Any]:
    edge_count0 = int(sample.unique_edges.shape[0])
    slot_levels: list[list[torch.Tensor]] = []
    line_adj_levels: list[torch.Tensor] = []
    pool_maps: list[torch.Tensor] = []
    unpool_maps: list[torch.Tensor] = []
    edge_counts: list[int] = [edge_count0]

    current_slots = build_slot_matrices(sample.unique_edges, sample.faces)
    slot_levels.append(list(current_slots))
    line_adj_levels.append(build_line_adjacency(current_slots))

    for ratio in pool_ratios:
        current_edges = int(current_slots[0].shape[0])
        target_edges = int(math.ceil(edge_count0 * float(ratio)))
        target_edges = min(current_edges, max(int(min_edges_per_level), target_edges))
        pool, unpool, current_slots = coarsen_slot_matrices(current_slots, target_edges, getattr(sample, 'vertices', None))
        pool_maps.append(pool)
        unpool_maps.append(unpool)
        edge_counts.append(int(pool.shape[0]))
        slot_levels.append(list(current_slots))
        line_adj_levels.append(build_line_adjacency(current_slots))

    cache: dict[str, Any] = {
        'slot_adj_levels': slot_levels,
        'line_adj_levels': line_adj_levels,
        'pool_maps': pool_maps,
        'unpool_maps': unpool_maps,
        'edge_counts': edge_counts,
        'config': {
            'pool_ratios': tuple(float(r) for r in pool_ratios),
            'min_edges_per_level': int(min_edges_per_level),
            'edge_count': edge_count0,
        },
    }
    sample.sparse_cache = cache
    assert_sparse_cache_cpu_only(cache)
    return cache


def move_sparse_tensor(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    if tensor.layout == torch.sparse_coo:
        return tensor.coalesce().to(device=device, copy=True).coalesce()
    return tensor.to(device=device, copy=True)


def _materialize_entry_for_step(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return move_sparse_tensor(value, device)
    if isinstance(value, dict):
        return {key: _materialize_entry_for_step(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize_entry_for_step(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_entry_for_step(item, device) for item in value)
    return value


def materialize_sparse_cache_for_step(cpu_cache: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    assert_sparse_cache_cpu_only(cpu_cache)
    return _materialize_entry_for_step(cpu_cache, device)


def get_or_build_sparse_cache(
    sample: MeshCNNSample,
    pool_ratios: tuple[float, ...] = (0.80, 0.64, 0.48),
    min_edges_per_level: int = 32,
) -> dict[str, Any]:
    expected = {
        'pool_ratios': tuple(float(r) for r in pool_ratios),
        'min_edges_per_level': int(min_edges_per_level),
        'edge_count': int(sample.unique_edges.shape[0]),
    }
    cache = getattr(sample, 'sparse_cache', None)
    if not isinstance(cache, dict) or cache.get('config') != expected:
        cache = build_sparse_cache(sample, pool_ratios=pool_ratios, min_edges_per_level=min_edges_per_level)
    else:
        _drop_legacy_device_cache_entries(cache)

    assert_sparse_cache_cpu_only(cache)
    return cache
