from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


_PROB_EPS = 1e-6


@dataclass(frozen=True)
class SeamComponentState:
    edge_component_ids: np.ndarray
    component_sizes: np.ndarray
    vertex_degrees: dict[int, int]
    terminals: tuple[int, ...]


@dataclass(frozen=True)
class GapCandidate:
    start_terminal: int
    end_terminal: int
    start_component: int
    end_component: int
    vertex_path: tuple[int, ...]
    edge_path: tuple[int, ...]
    added_edges: tuple[int, ...]
    hop_length: int
    path_cost: float


@dataclass(frozen=True)
class SeamPostprocessResult:
    initial_mask: np.ndarray
    gap_closed_mask: np.ndarray
    final_mask: np.ndarray
    closed_paths: tuple[tuple[int, ...], ...]
    added_edge_indices: tuple[int, ...]
    pruned_edge_indices: tuple[int, ...]
    terminal_count_before: int
    terminal_count_after_gap_closing: int
    pruned_component_count: int


def _canonical_edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _as_probability_array(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not np.isfinite(probs).all():
        raise ValueError('probabilities must be finite')
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError('probabilities must lie in [0, 1]')
    return probs


def _as_unique_edges(unique_edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(unique_edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f'unique_edges must have shape [E, 2], got {edges.shape}')
    return edges


def _validate_topology(topology: Any, edge_count: int) -> None:
    if topology is None:
        return
    for attr in ('canonical_edges', 'unique_edges'):
        value = getattr(topology, attr, None)
        if value is None:
            continue
        if len(value) != edge_count:
            raise ValueError(
                f'topology {attr} length {len(value)} does not match unique_edges length {edge_count}'
            )
        return


def _build_vertex_incidence(unique_edges: np.ndarray) -> tuple[dict[int, list[int]], dict[tuple[int, int], int]]:
    vertex_to_edges: dict[int, list[int]] = {}
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_idx, edge in enumerate(unique_edges):
        vi, vj = int(edge[0]), int(edge[1])
        if vi == vj:
            raise ValueError(f'degenerate edge at index {edge_idx}: {(vi, vj)}')
        edge_lookup[_canonical_edge_key(vi, vj)] = int(edge_idx)
        vertex_to_edges.setdefault(vi, []).append(int(edge_idx))
        vertex_to_edges.setdefault(vj, []).append(int(edge_idx))
    return vertex_to_edges, edge_lookup


def _build_vertex_graphs(unique_edges: np.ndarray, probabilities: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
    vertex_count = int(unique_edges.max()) + 1 if len(unique_edges) else 0
    if vertex_count == 0:
        empty = csr_matrix((0, 0), dtype=np.float64)
        return empty, empty

    rows = np.empty(len(unique_edges) * 2, dtype=np.int64)
    cols = np.empty(len(unique_edges) * 2, dtype=np.int64)
    weighted = np.empty(len(unique_edges) * 2, dtype=np.float64)
    unweighted = np.ones(len(unique_edges) * 2, dtype=np.float64)
    costs = -np.log(np.clip(probabilities, _PROB_EPS, 1.0))

    for edge_idx, edge in enumerate(unique_edges):
        vi, vj = int(edge[0]), int(edge[1])
        base = edge_idx * 2
        rows[base:base + 2] = (vi, vj)
        cols[base:base + 2] = (vj, vi)
        weighted[base:base + 2] = costs[edge_idx]

    weighted_graph = csr_matrix((weighted, (rows, cols)), shape=(vertex_count, vertex_count))
    unweighted_graph = csr_matrix((unweighted, (rows, cols)), shape=(vertex_count, vertex_count))
    return weighted_graph, unweighted_graph


def _component_state(mask: np.ndarray, unique_edges: np.ndarray, vertex_to_edges: dict[int, list[int]]) -> SeamComponentState:
    edge_component_ids = np.full(len(unique_edges), -1, dtype=np.int64)
    component_sizes: list[int] = []
    vertex_degrees: dict[int, int] = {}

    seam_indices = np.flatnonzero(mask)
    for edge_idx in seam_indices:
        vi, vj = int(unique_edges[edge_idx, 0]), int(unique_edges[edge_idx, 1])
        vertex_degrees[vi] = vertex_degrees.get(vi, 0) + 1
        vertex_degrees[vj] = vertex_degrees.get(vj, 0) + 1

    component_id = 0
    for edge_idx in seam_indices:
        edge_idx = int(edge_idx)
        if edge_component_ids[edge_idx] >= 0:
            continue
        queue = deque([edge_idx])
        members: list[int] = []
        edge_component_ids[edge_idx] = component_id

        while queue:
            current = queue.popleft()
            members.append(current)
            vi, vj = int(unique_edges[current, 0]), int(unique_edges[current, 1])
            for vertex in (vi, vj):
                for neighbor in vertex_to_edges.get(vertex, ()):
                    if not mask[neighbor] or edge_component_ids[neighbor] >= 0:
                        continue
                    edge_component_ids[neighbor] = component_id
                    queue.append(int(neighbor))

        component_sizes.append(len(members))
        component_id += 1

    terminals = tuple(sorted(vertex for vertex, degree in vertex_degrees.items() if degree == 1))
    return SeamComponentState(
        edge_component_ids=edge_component_ids,
        component_sizes=np.asarray(component_sizes, dtype=np.int64),
        vertex_degrees=vertex_degrees,
        terminals=terminals,
    )


def _component_id_for_terminal(vertex: int, mask: np.ndarray, vertex_to_edges: dict[int, list[int]], component_ids: np.ndarray) -> int:
    for edge_idx in vertex_to_edges.get(vertex, ()):
        if mask[edge_idx]:
            return int(component_ids[edge_idx])
    return -1


def _reconstruct_vertex_path(start: int, target: int, predecessors: np.ndarray) -> tuple[int, ...] | None:
    path = [int(target)]
    current = int(target)
    while current != start:
        current = int(predecessors[current])
        if current < 0:
            return None
        path.append(current)
    path.reverse()
    return tuple(path)


def _edge_path_from_vertices(vertex_path: tuple[int, ...], edge_lookup: dict[tuple[int, int], int]) -> tuple[int, ...]:
    edges = []
    for idx in range(len(vertex_path) - 1):
        key = _canonical_edge_key(int(vertex_path[idx]), int(vertex_path[idx + 1]))
        edges.append(int(edge_lookup[key]))
    return tuple(edges)


def _enumerate_gap_candidates(
    *,
    mask: np.ndarray,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    vertex_to_edges: dict[int, list[int]],
    edge_lookup: dict[tuple[int, int], int],
    weighted_graph: csr_matrix,
    unweighted_graph: csr_matrix,
    max_gap_length: int,
) -> list[GapCandidate]:
    if max_gap_length < 1:
        return []

    state = _component_state(mask, unique_edges, vertex_to_edges)
    terminals = state.terminals
    if len(terminals) < 2:
        return []

    costs = -np.log(np.clip(probabilities, _PROB_EPS, 1.0))
    candidates: list[GapCandidate] = []

    for start_idx, start_terminal in enumerate(terminals):
        start_component = _component_id_for_terminal(
            start_terminal,
            mask,
            vertex_to_edges,
            state.edge_component_ids,
        )
        if start_component < 0:
            continue

        hop_distances = dijkstra(
            unweighted_graph,
            directed=False,
            indices=int(start_terminal),
            unweighted=True,
            limit=float(max_gap_length),
        )
        weighted_distances, predecessors = dijkstra(
            weighted_graph,
            directed=False,
            indices=int(start_terminal),
            return_predecessors=True,
        )

        for end_terminal in terminals[start_idx + 1:]:
            end_component = _component_id_for_terminal(
                end_terminal,
                mask,
                vertex_to_edges,
                state.edge_component_ids,
            )
            if end_component < 0 or end_component == start_component:
                continue
            if not np.isfinite(hop_distances[end_terminal]) or hop_distances[end_terminal] > max_gap_length:
                continue

            vertex_path = _reconstruct_vertex_path(int(start_terminal), int(end_terminal), predecessors)
            if vertex_path is None or len(vertex_path) < 2:
                continue
            if len(vertex_path) - 1 > max_gap_length:
                continue

            interior_vertices = vertex_path[1:-1]
            if any(state.vertex_degrees.get(int(vertex), 0) > 0 for vertex in interior_vertices):
                continue

            edge_path = _edge_path_from_vertices(vertex_path, edge_lookup)
            added_edges = tuple(edge_idx for edge_idx in edge_path if not mask[edge_idx])
            if not added_edges:
                continue

            candidates.append(GapCandidate(
                start_terminal=int(start_terminal),
                end_terminal=int(end_terminal),
                start_component=int(start_component),
                end_component=int(end_component),
                vertex_path=vertex_path,
                edge_path=edge_path,
                added_edges=added_edges,
                hop_length=int(len(edge_path)),
                path_cost=float(sum(costs[edge_idx] for edge_idx in edge_path)),
            ))

    candidates.sort(
        key=lambda candidate: (
            len(candidate.added_edges),
            candidate.path_cost,
            candidate.hop_length,
            candidate.start_terminal,
            candidate.end_terminal,
            candidate.added_edges,
        )
    )
    return candidates


def _prune_small_components(
    mask: np.ndarray,
    unique_edges: np.ndarray,
    vertex_to_edges: dict[int, list[int]],
    min_island_size: int,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    if min_island_size <= 1:
        return mask.copy(), (), 0

    state = _component_state(mask, unique_edges, vertex_to_edges)
    if state.component_sizes.size == 0:
        return mask.copy(), (), 0

    keep = np.ones(len(mask), dtype=bool)
    pruned_edges: list[int] = []
    pruned_components = 0
    for component_id, size in enumerate(state.component_sizes):
        if int(size) >= min_island_size:
            continue
        component_edges = np.flatnonzero(state.edge_component_ids == component_id)
        if component_edges.size == 0:
            continue
        keep[component_edges] = False
        pruned_edges.extend(int(edge_idx) for edge_idx in component_edges)
        pruned_components += 1

    out = mask.copy()
    out[~keep] = False
    return out, tuple(sorted(pruned_edges)), pruned_components


def apply_seam_postprocessing_detailed(
    topology: Any,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    max_gap_length: int = 5,
    min_island_size: int = 3,
) -> SeamPostprocessResult:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f'threshold must be in [0, 1], got {threshold}')
    if max_gap_length < 0:
        raise ValueError(f'max_gap_length must be non-negative, got {max_gap_length}')
    if min_island_size < 1:
        raise ValueError(f'min_island_size must be at least 1, got {min_island_size}')

    probs = _as_probability_array(probabilities)
    edges = _as_unique_edges(unique_edges)
    if len(probs) != len(edges):
        raise ValueError(
            f'probabilities length {len(probs)} does not match unique_edges length {len(edges)}'
        )
    _validate_topology(topology, len(edges))

    initial_mask = probs >= float(threshold)
    vertex_to_edges, edge_lookup = _build_vertex_incidence(edges)
    weighted_graph, unweighted_graph = _build_vertex_graphs(edges, probs)
    terminal_count_before = len(_component_state(initial_mask, edges, vertex_to_edges).terminals)

    gap_closed_mask = initial_mask.copy()
    closed_paths: list[tuple[int, ...]] = []
    while True:
        candidates = _enumerate_gap_candidates(
            mask=gap_closed_mask,
            unique_edges=edges,
            probabilities=probs,
            vertex_to_edges=vertex_to_edges,
            edge_lookup=edge_lookup,
            weighted_graph=weighted_graph,
            unweighted_graph=unweighted_graph,
            max_gap_length=max_gap_length,
        )
        if not candidates:
            break
        best = candidates[0]
        gap_closed_mask[np.asarray(best.added_edges, dtype=np.int64)] = True
        closed_paths.append(best.added_edges)

    terminal_count_after_gap_closing = len(_component_state(gap_closed_mask, edges, vertex_to_edges).terminals)
    final_mask, pruned_edge_indices, pruned_component_count = _prune_small_components(
        gap_closed_mask,
        edges,
        vertex_to_edges,
        min_island_size,
    )

    initial_indices = set(int(idx) for idx in np.flatnonzero(initial_mask))
    final_indices = tuple(sorted(int(idx) for idx in np.flatnonzero(final_mask)))
    added_edge_indices = tuple(sorted(set(final_indices) - initial_indices))

    return SeamPostprocessResult(
        initial_mask=initial_mask,
        gap_closed_mask=gap_closed_mask,
        final_mask=final_mask,
        closed_paths=tuple(tuple(int(edge_idx) for edge_idx in path) for path in closed_paths),
        added_edge_indices=added_edge_indices,
        pruned_edge_indices=pruned_edge_indices,
        terminal_count_before=int(terminal_count_before),
        terminal_count_after_gap_closing=int(terminal_count_after_gap_closing),
        pruned_component_count=int(pruned_component_count),
    )


def apply_seam_postprocessing(
    topology: Any,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    max_gap_length: int = 5,
    min_island_size: int = 3,
) -> np.ndarray:
    return apply_seam_postprocessing_detailed(
        topology=topology,
        unique_edges=unique_edges,
        probabilities=probabilities,
        threshold=threshold,
        max_gap_length=max_gap_length,
        min_island_size=min_island_size,
    ).final_mask


def threshold_and_clean(
    probs: np.ndarray,
    unique_edges: np.ndarray,
    threshold: float = 0.5,
    min_component_size: int = 3,
) -> np.ndarray:
    return apply_seam_postprocessing(
        topology=None,
        unique_edges=unique_edges,
        probabilities=probs,
        threshold=threshold,
        max_gap_length=0,
        min_island_size=min_component_size,
    )


def stitch_seam_gaps(
    probs: np.ndarray,
    seam_mask: np.ndarray,
    unique_edges: np.ndarray,
    edge_to_faces: dict | None = None,
    max_gap: int = 3,
) -> np.ndarray:
    del edge_to_faces
    probs = _as_probability_array(probs)
    unique_edges = _as_unique_edges(unique_edges)
    seam_mask = np.asarray(seam_mask, dtype=bool).reshape(-1)
    if len(seam_mask) != len(unique_edges):
        raise ValueError('seam_mask length must match unique_edges length')

    seeded_probs = probs.copy()
    seeded_probs[seam_mask] = 1.0
    details = apply_seam_postprocessing_detailed(
        topology=None,
        unique_edges=unique_edges,
        probabilities=seeded_probs,
        threshold=0.5,
        max_gap_length=max_gap,
        min_island_size=1,
    )
    return details.gap_closed_mask


def postprocess_seams(
    probs: np.ndarray,
    unique_edges: np.ndarray,
    edge_to_faces: dict | None = None,
    threshold: float = 0.5,
    min_component_size: int = 3,
    max_gap: int = 3,
) -> np.ndarray:
    del edge_to_faces
    return apply_seam_postprocessing(
        topology=None,
        unique_edges=unique_edges,
        probabilities=probs,
        threshold=threshold,
        max_gap_length=max_gap,
        min_island_size=min_component_size,
    )
