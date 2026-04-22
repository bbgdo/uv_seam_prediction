from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree


_PROB_EPS = 1e-6
_DEFAULT_SKELETON_THRESHOLD = 0.2
_DEFAULT_SKELETON_RADIUS = 3


@dataclass(frozen=True)
class SeamPostprocessResult:
    threshold_mask: np.ndarray
    skeleton_mask: np.ndarray
    steiner_mask: np.ndarray
    final_mask: np.ndarray
    skeleton_deleted_vertices: tuple[int, ...]
    steiner_added_edges: tuple[int, ...]
    pruned_edge_indices: tuple[int, ...]
    skeleton_terminal_vertex_count: int
    steiner_terminal_group_count: int
    steiner_tree_count: int
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
        break


def _build_incidence(unique_edges: np.ndarray) -> tuple[dict[int, list[int]], dict[tuple[int, int], int]]:
    vertex_to_edges: dict[int, list[int]] = {}
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_idx, edge in enumerate(unique_edges):
        vi, vj = int(edge[0]), int(edge[1])
        if vi == vj:
            raise ValueError(f'degenerate edge at index {edge_idx}: {(vi, vj)}')
        vertex_to_edges.setdefault(vi, []).append(int(edge_idx))
        vertex_to_edges.setdefault(vj, []).append(int(edge_idx))
        edge_lookup[_canonical_edge_key(vi, vj)] = int(edge_idx)
    return vertex_to_edges, edge_lookup


def _build_vertex_probability(probabilities: np.ndarray, vertex_to_edges: dict[int, list[int]]) -> np.ndarray:
    vertex_count = max(vertex_to_edges.keys(), default=-1) + 1
    values = np.zeros(vertex_count, dtype=np.float64)
    for vertex, edge_indices in vertex_to_edges.items():
        values[int(vertex)] = float(np.max(probabilities[np.asarray(edge_indices, dtype=np.int64)]))
    return values


def _build_candidate_vertex_graph(unique_edges: np.ndarray, candidate_vertices: np.ndarray) -> dict[int, set[int]]:
    adjacency = {int(vertex): set() for vertex in np.flatnonzero(candidate_vertices)}
    for edge in unique_edges:
        vi, vj = int(edge[0]), int(edge[1])
        if not candidate_vertices[vi] or not candidate_vertices[vj]:
            continue
        adjacency.setdefault(vi, set()).add(vj)
        adjacency.setdefault(vj, set()).add(vi)
    return adjacency


def _connected_vertex_components(adjacency: dict[int, set[int]]) -> list[set[int]]:
    components: list[set[int]] = []
    remaining = set(adjacency.keys())
    while remaining:
        start = min(remaining)
        queue = deque([start])
        component = {start}
        remaining.remove(start)
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency.get(vertex, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _component_is_connected(component: set[int], adjacency: dict[int, set[int]]) -> bool:
    if len(component) <= 1:
        return True
    start = min(component)
    queue = deque([start])
    visited = {start}
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency.get(vertex, ()):
            if neighbor in component and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited) == len(component)


def _distances_to_active(
    original_component: set[int],
    active_component: set[int],
    adjacency: dict[int, set[int]],
) -> dict[int, int]:
    if not active_component:
        return {vertex: np.iinfo(np.int64).max for vertex in original_component}
    queue = deque(sorted(active_component))
    distances = {vertex: -1 for vertex in original_component}
    for vertex in active_component:
        distances[vertex] = 0
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency.get(vertex, ()):
            if neighbor not in original_component or distances[neighbor] >= 0:
                continue
            distances[neighbor] = distances[vertex] + 1
            queue.append(neighbor)
    return distances


def _skeletonize_vertices(
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    *,
    skeleton_threshold: float,
    skeleton_radius: int,
    threshold_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if skeleton_radius < 1:
        raise ValueError(f'skeleton_radius must be at least 1, got {skeleton_radius}')

    vertex_to_edges, _edge_lookup = _build_incidence(unique_edges)
    vertex_probabilities = _build_vertex_probability(probabilities, vertex_to_edges)
    threshold_vertices = np.zeros_like(vertex_probabilities, dtype=bool)
    for edge_idx in np.flatnonzero(threshold_mask):
        vi, vj = (int(unique_edges[edge_idx, 0]), int(unique_edges[edge_idx, 1]))
        threshold_vertices[vi] = True
        threshold_vertices[vj] = True

    candidate_vertices = (vertex_probabilities >= skeleton_threshold) & threshold_vertices
    adjacency = _build_candidate_vertex_graph(unique_edges, candidate_vertices)
    active = set(adjacency.keys())
    deleted_vertices: list[int] = []

    for original_component in _connected_vertex_components(adjacency):
        component_active = set(original_component)
        changed = True
        while changed:
            changed = False
            ordered = sorted(component_active, key=lambda vertex: (vertex_probabilities[vertex], vertex))
            for vertex in ordered:
                current_neighbors = adjacency[vertex] & component_active
                if len(current_neighbors) <= 1:
                    continue
                reduced_component = set(component_active)
                reduced_component.remove(vertex)
                if not _component_is_connected(reduced_component, adjacency):
                    continue
                distances = _distances_to_active(original_component, reduced_component, adjacency)
                if max(distances.values(), default=0) > skeleton_radius:
                    continue
                component_active = reduced_component
                deleted_vertices.append(int(vertex))
                active.remove(int(vertex))
                changed = True
                break

    active_mask = np.zeros_like(vertex_probabilities, dtype=bool)
    if active:
        active_mask[np.asarray(sorted(active), dtype=np.int64)] = True
    return active_mask, tuple(sorted(deleted_vertices))


def _mask_from_active_vertices(unique_edges: np.ndarray, threshold_mask: np.ndarray, active_vertices: np.ndarray) -> np.ndarray:
    skeleton_mask = np.zeros(len(unique_edges), dtype=bool)
    for edge_idx in np.flatnonzero(threshold_mask):
        vi, vj = int(unique_edges[edge_idx, 0]), int(unique_edges[edge_idx, 1])
        if active_vertices[vi] and active_vertices[vj]:
            skeleton_mask[int(edge_idx)] = True
    return skeleton_mask


def _residual_components(vertex_count: int, unique_edges: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    rows: list[int] = []
    cols: list[int] = []
    for edge_idx, edge in enumerate(unique_edges):
        if seam_mask[edge_idx]:
            continue
        vi, vj = int(edge[0]), int(edge[1])
        rows.extend((vi, vj))
        cols.extend((vj, vi))
    data = np.ones(len(rows), dtype=np.float64)
    graph = csr_matrix((data, (rows, cols)), shape=(vertex_count, vertex_count))
    if vertex_count == 0:
        return np.zeros(0, dtype=np.int64)
    _component_count, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int64, copy=False)


def _terminal_groups_by_residual_component(
    unique_edges: np.ndarray,
    seam_mask: np.ndarray,
    residual_labels: np.ndarray,
) -> list[tuple[int, ...]]:
    groups: dict[int, set[int]] = {}
    for edge_idx in np.flatnonzero(seam_mask):
        vi, vj = int(unique_edges[edge_idx, 0]), int(unique_edges[edge_idx, 1])
        groups.setdefault(int(residual_labels[vi]), set()).add(vi)
        groups.setdefault(int(residual_labels[vj]), set()).add(vj)
    return [tuple(sorted(vertices)) for vertices in groups.values() if len(vertices) >= 2]


def _edge_costs(
    probabilities: np.ndarray,
    *,
    distortion_per_edge: np.ndarray | None,
    confidence_weight: float,
    distortion_weight: float,
) -> np.ndarray:
    confidence_cost = -np.log(np.clip(probabilities, _PROB_EPS, 1.0))
    if distortion_per_edge is None:
        return confidence_cost

    distortion = np.asarray(distortion_per_edge, dtype=np.float64).reshape(-1)
    if distortion.shape != probabilities.shape:
        raise ValueError('distortion_per_edge must match probabilities shape')
    if not np.isfinite(distortion).all():
        raise ValueError('distortion_per_edge must be finite')
    distortion_cost = np.clip(1.0 - np.abs(distortion), 0.0, None)
    return confidence_weight * confidence_cost + distortion_weight * distortion_cost


def _build_residual_graph(
    vertex_count: int,
    unique_edges: np.ndarray,
    seam_mask: np.ndarray,
    edge_costs: np.ndarray,
) -> tuple[csr_matrix, csr_matrix]:
    rows: list[int] = []
    cols: list[int] = []
    weighted: list[float] = []
    hops: list[float] = []
    for edge_idx, edge in enumerate(unique_edges):
        if seam_mask[edge_idx]:
            continue
        vi, vj = int(edge[0]), int(edge[1])
        rows.extend((vi, vj))
        cols.extend((vj, vi))
        weight = float(edge_costs[edge_idx])
        weighted.extend((weight, weight))
        hops.extend((1.0, 1.0))
    weighted_graph = csr_matrix((np.asarray(weighted, dtype=np.float64), (rows, cols)), shape=(vertex_count, vertex_count))
    hop_graph = csr_matrix((np.asarray(hops, dtype=np.float64), (rows, cols)), shape=(vertex_count, vertex_count))
    return weighted_graph, hop_graph


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


def _vertex_path_to_edges(vertex_path: tuple[int, ...], edge_lookup: dict[tuple[int, int], int]) -> tuple[int, ...]:
    edges = []
    for idx in range(len(vertex_path) - 1):
        edges.append(int(edge_lookup[_canonical_edge_key(int(vertex_path[idx]), int(vertex_path[idx + 1]))]))
    return tuple(edges)


def _steiner_edges_for_terminals(
    terminals: tuple[int, ...],
    *,
    weighted_graph: csr_matrix,
    hop_graph: csr_matrix,
    edge_lookup: dict[tuple[int, int], int],
    max_path_hops: int,
) -> tuple[int, ...]:
    terminal_indices = np.asarray(terminals, dtype=np.int64)
    weighted_distances, predecessors = dijkstra(
        weighted_graph,
        directed=False,
        indices=terminal_indices,
        return_predecessors=True,
    )
    hop_distances = dijkstra(
        hop_graph,
        directed=False,
        indices=terminal_indices,
        unweighted=True,
    )

    metric = np.full((len(terminals), len(terminals)), np.inf, dtype=np.float64)
    for i in range(len(terminals)):
        for j in range(i + 1, len(terminals)):
            hops = hop_distances[i, terminals[j]]
            if not np.isfinite(hops):
                continue
            if max_path_hops > 0 and hops > max_path_hops:
                continue
            metric[i, j] = weighted_distances[i, terminals[j]]
            metric[j, i] = metric[i, j]

    closure = csr_matrix(metric)
    mst = minimum_spanning_tree(closure)
    mst_rows, mst_cols = mst.nonzero()
    added_edges: set[int] = set()
    for row, col in zip(mst_rows.tolist(), mst_cols.tolist()):
        start = int(terminals[row])
        end = int(terminals[col])
        vertex_path = _reconstruct_vertex_path(start, end, predecessors[row])
        if vertex_path is None:
            continue
        added_edges.update(_vertex_path_to_edges(vertex_path, edge_lookup))
    return tuple(sorted(added_edges))


def _steiner_connect_components(
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    skeleton_mask: np.ndarray,
    *,
    max_path_hops: int,
    distortion_per_edge: np.ndarray | None,
    confidence_weight: float,
    distortion_weight: float,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    vertex_count = int(unique_edges.max()) + 1 if len(unique_edges) else 0
    if vertex_count == 0:
        return np.zeros(len(unique_edges), dtype=bool), (), 0

    residual_labels = _residual_components(vertex_count, unique_edges, skeleton_mask)
    terminal_groups = _terminal_groups_by_residual_component(unique_edges, skeleton_mask, residual_labels)
    vertex_to_edges, edge_lookup = _build_incidence(unique_edges)
    del vertex_to_edges
    edge_costs = _edge_costs(
        probabilities,
        distortion_per_edge=distortion_per_edge,
        confidence_weight=confidence_weight,
        distortion_weight=distortion_weight,
    )
    weighted_graph, hop_graph = _build_residual_graph(vertex_count, unique_edges, skeleton_mask, edge_costs)

    added_edges: set[int] = set()
    tree_count = 0
    for terminals in terminal_groups:
        component_edges = _steiner_edges_for_terminals(
            terminals,
            weighted_graph=weighted_graph,
            hop_graph=hop_graph,
            edge_lookup=edge_lookup,
            max_path_hops=max_path_hops,
        )
        if component_edges:
            added_edges.update(component_edges)
            tree_count += 1

    steiner_mask = np.zeros(len(unique_edges), dtype=bool)
    if added_edges:
        steiner_mask[np.asarray(sorted(added_edges), dtype=np.int64)] = True
    return steiner_mask, tuple(sorted(added_edges)), tree_count


def _component_edge_labels(mask: np.ndarray, unique_edges: np.ndarray, vertex_to_edges: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(len(unique_edges), -1, dtype=np.int64)
    sizes: list[int] = []
    component_id = 0
    for edge_idx in np.flatnonzero(mask):
        edge_idx = int(edge_idx)
        if labels[edge_idx] >= 0:
            continue
        queue = deque([edge_idx])
        labels[edge_idx] = component_id
        members = 0
        while queue:
            current = queue.popleft()
            members += 1
            vi, vj = int(unique_edges[current, 0]), int(unique_edges[current, 1])
            for vertex in (vi, vj):
                for neighbor in vertex_to_edges.get(vertex, ()):
                    if not mask[neighbor] or labels[neighbor] >= 0:
                        continue
                    labels[neighbor] = component_id
                    queue.append(int(neighbor))
        sizes.append(members)
        component_id += 1
    return labels, np.asarray(sizes, dtype=np.int64)


def _prune_small_components(mask: np.ndarray, unique_edges: np.ndarray, min_island_size: int) -> tuple[np.ndarray, tuple[int, ...], int]:
    if min_island_size <= 1:
        return mask.copy(), (), 0
    vertex_to_edges, _edge_lookup = _build_incidence(unique_edges)
    labels, sizes = _component_edge_labels(mask, unique_edges, vertex_to_edges)
    if sizes.size == 0:
        return mask.copy(), (), 0

    pruned_edges: list[int] = []
    pruned_components = 0
    out = mask.copy()
    for component_id, size in enumerate(sizes):
        if int(size) >= min_island_size:
            continue
        component_edges = np.flatnonzero(labels == component_id)
        if component_edges.size == 0:
            continue
        out[component_edges] = False
        pruned_edges.extend(int(edge_idx) for edge_idx in component_edges)
        pruned_components += 1
    return out, tuple(sorted(pruned_edges)), pruned_components


def apply_seam_postprocessing_detailed(
    topology: Any,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    max_gap_length: int = 5,
    min_island_size: int = 3,
    *,
    skeleton_threshold: float = _DEFAULT_SKELETON_THRESHOLD,
    skeleton_radius: int = _DEFAULT_SKELETON_RADIUS,
    distortion_per_edge: np.ndarray | None = None,
    confidence_weight: float = 1.0,
    distortion_weight: float = 1.0,
) -> SeamPostprocessResult:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f'threshold must be in [0, 1], got {threshold}')
    if skeleton_threshold < 0.0 or skeleton_threshold > 1.0:
        raise ValueError(f'skeleton_threshold must be in [0, 1], got {skeleton_threshold}')
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

    threshold_mask = probs >= float(threshold)
    active_vertices, deleted_vertices = _skeletonize_vertices(
        edges,
        probs,
        skeleton_threshold=float(min(skeleton_threshold, threshold) if threshold_mask.any() else skeleton_threshold),
        skeleton_radius=int(skeleton_radius),
        threshold_mask=threshold_mask,
    )
    skeleton_mask = _mask_from_active_vertices(edges, threshold_mask, active_vertices)

    steiner_mask, steiner_added_edges, steiner_tree_count = _steiner_connect_components(
        edges,
        probs,
        skeleton_mask,
        max_path_hops=int(max_gap_length),
        distortion_per_edge=distortion_per_edge,
        confidence_weight=float(confidence_weight),
        distortion_weight=float(distortion_weight),
    )
    combined_mask = skeleton_mask | steiner_mask
    final_mask, pruned_edge_indices, pruned_component_count = _prune_small_components(
        combined_mask,
        edges,
        int(min_island_size),
    )

    skeleton_terminal_vertices = set()
    residual_labels = _residual_components(int(edges.max()) + 1 if len(edges) else 0, edges, skeleton_mask)
    terminal_groups = _terminal_groups_by_residual_component(edges, skeleton_mask, residual_labels) if len(edges) else []
    for edge_idx in np.flatnonzero(skeleton_mask):
        skeleton_terminal_vertices.update((int(edges[edge_idx, 0]), int(edges[edge_idx, 1])))

    return SeamPostprocessResult(
        threshold_mask=threshold_mask,
        skeleton_mask=skeleton_mask,
        steiner_mask=steiner_mask,
        final_mask=final_mask,
        skeleton_deleted_vertices=deleted_vertices,
        steiner_added_edges=steiner_added_edges,
        pruned_edge_indices=pruned_edge_indices,
        skeleton_terminal_vertex_count=int(len(skeleton_terminal_vertices)),
        steiner_terminal_group_count=int(len(terminal_groups)),
        steiner_tree_count=int(steiner_tree_count),
        pruned_component_count=int(pruned_component_count),
    )


def apply_seam_postprocessing(
    topology: Any,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    max_gap_length: int = 5,
    min_island_size: int = 3,
    **kwargs: Any,
) -> np.ndarray:
    return apply_seam_postprocessing_detailed(
        topology=topology,
        unique_edges=unique_edges,
        probabilities=probabilities,
        threshold=threshold,
        max_gap_length=max_gap_length,
        min_island_size=min_island_size,
        **kwargs,
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
    probabilities = _as_probability_array(probs).copy()
    seam_mask = np.asarray(seam_mask, dtype=bool).reshape(-1)
    probabilities[seam_mask] = 1.0
    return apply_seam_postprocessing(
        topology=None,
        unique_edges=unique_edges,
        probabilities=probabilities,
        threshold=0.5,
        max_gap_length=max_gap,
        min_island_size=1,
    )


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
