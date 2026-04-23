from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any

import numpy as np


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


@dataclass(frozen=True)
class _GraphViews:
    dual_neighbors: tuple[tuple[int, ...], ...]
    seam_neighbors: tuple[tuple[int, ...], ...]
    edge_lengths: np.ndarray
    edge_to_vertices: np.ndarray
    vertex_to_edges: tuple[tuple[int, ...], ...]
    edge_lookup: dict[tuple[int, int], int]
    vertex_count: int


@dataclass(frozen=True)
class _BridgeCandidate:
    endpoints: tuple[int, int]
    edge_indices: tuple[int, ...]
    total_cost: float
    total_length: float
    mean_confidence: float


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


def _validate_probability_threshold(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f'{name} must be in [0, 1], got {value}')


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


def _build_incidence(unique_edges: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], dict[tuple[int, int], int], int]:
    if np.any(unique_edges < 0):
        first = unique_edges[np.any(unique_edges < 0, axis=1)][0]
        raise ValueError(f'edge vertex ids must be non-negative, got {(int(first[0]), int(first[1]))}')
    vertex_count = int(unique_edges.max()) + 1 if len(unique_edges) else 0
    incident: list[list[int]] = [[] for _ in range(vertex_count)]
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_idx, edge in enumerate(unique_edges):
        vi, vj = int(edge[0]), int(edge[1])
        if vi == vj:
            raise ValueError(f'degenerate edge at index {edge_idx}: {(vi, vj)}')
        key = _canonical_edge_key(vi, vj)
        if key in edge_lookup:
            raise ValueError(f'duplicate edge {key}')
        edge_lookup[key] = int(edge_idx)
        incident[vi].append(int(edge_idx))
        incident[vj].append(int(edge_idx))
    vertex_to_edges = tuple(tuple(sorted(edges)) for edges in incident)
    return vertex_to_edges, edge_lookup, vertex_count


def _build_seam_neighbors(vertex_to_edges: tuple[tuple[int, ...], ...], edge_count: int) -> tuple[tuple[int, ...], ...]:
    neighbors = [set() for _ in range(edge_count)]
    for incident in vertex_to_edges:
        for i, edge_a in enumerate(incident):
            for edge_b in incident[i + 1:]:
                neighbors[edge_a].add(edge_b)
                neighbors[edge_b].add(edge_a)
    return tuple(tuple(sorted(values)) for values in neighbors)


def _faces_from_topology(topology: Any) -> np.ndarray | None:
    if topology is None:
        return None
    canonical_faces = getattr(topology, 'canonical_faces', None)
    if canonical_faces is not None:
        return np.asarray([face.vertex_ids for face in canonical_faces], dtype=np.int64).reshape((-1, 3))
    faces = getattr(topology, 'faces', None)
    if faces is None:
        return None
    faces_array = np.asarray(faces, dtype=np.int64)
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        return None
    return faces_array


def _infer_triangles_from_edges(
    unique_edges: np.ndarray,
    vertex_to_edges: tuple[tuple[int, ...], ...],
    edge_lookup: dict[tuple[int, int], int],
) -> np.ndarray:
    vertex_neighbors: list[set[int]] = [set() for _ in vertex_to_edges]
    for vi, vj in unique_edges:
        a = int(vi)
        b = int(vj)
        vertex_neighbors[a].add(b)
        vertex_neighbors[b].add(a)

    triangles: set[tuple[int, int, int]] = set()
    for vi, vj in unique_edges:
        a = int(vi)
        b = int(vj)
        for c in vertex_neighbors[a] & vertex_neighbors[b]:
            tri = tuple(sorted((a, b, int(c))))
            if len(set(tri)) == 3:
                triangles.add(tri)

    valid: list[tuple[int, int, int]] = []
    for a, b, c in sorted(triangles):
        if (
            _canonical_edge_key(a, b) in edge_lookup
            and _canonical_edge_key(b, c) in edge_lookup
            and _canonical_edge_key(a, c) in edge_lookup
        ):
            valid.append((a, b, c))
    return np.asarray(valid, dtype=np.int64).reshape((-1, 3))


def _build_dual_neighbors(
    unique_edges: np.ndarray,
    edge_lookup: dict[tuple[int, int], int],
    vertex_to_edges: tuple[tuple[int, ...], ...],
    topology: Any,
) -> tuple[tuple[int, ...], ...]:
    faces = _faces_from_topology(topology)
    if faces is None:
        faces = _infer_triangles_from_edges(unique_edges, vertex_to_edges, edge_lookup)

    neighbors = [set() for _ in range(len(unique_edges))]
    for face in faces:
        if len(set(int(vertex) for vertex in face)) != 3:
            continue
        edge_ids = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge_idx = edge_lookup.get(_canonical_edge_key(int(face[a]), int(face[b])))
            if edge_idx is None:
                edge_ids = []
                break
            edge_ids.append(edge_idx)
        for i, edge_a in enumerate(edge_ids):
            for edge_b in edge_ids[i + 1:]:
                neighbors[edge_a].add(edge_b)
                neighbors[edge_b].add(edge_a)
    return tuple(tuple(sorted(values)) for values in neighbors)


def _edge_lengths(unique_edges: np.ndarray, topology: Any) -> np.ndarray:
    if len(unique_edges) == 0:
        return np.zeros(0, dtype=np.float64)
    vertices = getattr(topology, 'canonical_vertices', None) if topology is not None else None
    if vertices is None:
        return np.ones(len(unique_edges), dtype=np.float64)
    coords = np.asarray(vertices, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return np.ones(len(unique_edges), dtype=np.float64)
    if int(unique_edges.max()) >= len(coords):
        return np.ones(len(unique_edges), dtype=np.float64)
    deltas = coords[unique_edges[:, 0]] - coords[unique_edges[:, 1]]
    lengths = np.linalg.norm(deltas, axis=1).astype(np.float64, copy=False)
    return np.where(lengths > 0.0, lengths, 1.0)


def _build_graph_views(topology: Any, unique_edges: np.ndarray) -> _GraphViews:
    vertex_to_edges, edge_lookup, vertex_count = _build_incidence(unique_edges)
    return _GraphViews(
        dual_neighbors=_build_dual_neighbors(unique_edges, edge_lookup, vertex_to_edges, topology),
        seam_neighbors=_build_seam_neighbors(vertex_to_edges, len(unique_edges)),
        edge_lengths=_edge_lengths(unique_edges, topology),
        edge_to_vertices=unique_edges.astype(np.int64, copy=True),
        vertex_to_edges=vertex_to_edges,
        edge_lookup=edge_lookup,
        vertex_count=vertex_count,
    )


def _smooth_probabilities(
    probabilities: np.ndarray,
    dual_neighbors: tuple[tuple[int, ...], ...],
    *,
    beta: float,
    iterations: int,
) -> np.ndarray:
    if iterations == 0 or len(probabilities) == 0:
        return probabilities.copy()
    current = probabilities.copy()
    for _ in range(iterations):
        smoothed = probabilities.copy()
        for edge_idx, neighbors in enumerate(dual_neighbors):
            if neighbors:
                smoothed[edge_idx] = (1.0 - beta) * probabilities[edge_idx] + beta * float(np.mean(current[list(neighbors)]))
        current = smoothed
    return np.clip(current, 0.0, 1.0)


def _component_members(mask: np.ndarray, neighbors: tuple[tuple[int, ...], ...]) -> list[tuple[int, ...]]:
    visited = np.zeros(len(mask), dtype=bool)
    components: list[tuple[int, ...]] = []
    for start in np.flatnonzero(mask):
        start_idx = int(start)
        if visited[start_idx]:
            continue
        queue = deque([start_idx])
        visited[start_idx] = True
        members: list[int] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in neighbors[current]:
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(int(neighbor))
        components.append(tuple(sorted(members)))
    return components


def _hysteresis_threshold(refined: np.ndarray, neighbors: tuple[tuple[int, ...], ...], tau_low: float, tau_high: float) -> np.ndarray:
    strong = refined >= tau_high
    weak_or_strong = refined >= tau_low
    if not np.any(strong):
        return np.zeros(len(refined), dtype=bool)
    if np.all(strong):
        return strong.copy()

    out = np.zeros(len(refined), dtype=bool)
    for component in _component_members(weak_or_strong, neighbors):
        component_idx = np.asarray(component, dtype=np.int64)
        if bool(np.any(strong[component_idx])):
            out[component_idx] = True
    return out


def _active_degrees(component: tuple[int, ...], mask: np.ndarray, neighbors: tuple[tuple[int, ...], ...]) -> dict[int, int]:
    component_set = set(component)
    return {
        edge_idx: sum(1 for neighbor in neighbors[edge_idx] if mask[neighbor] and neighbor in component_set)
        for edge_idx in component
    }


def _component_has_cycle(component: tuple[int, ...], mask: np.ndarray, neighbors: tuple[tuple[int, ...], ...]) -> bool:
    adjacency_edges = 0
    component_set = set(component)
    for edge_idx in component:
        adjacency_edges += sum(1 for neighbor in neighbors[edge_idx] if mask[neighbor] and neighbor in component_set)
    return adjacency_edges // 2 >= len(component)


def _prune_tiny_components(
    mask: np.ndarray,
    refined: np.ndarray,
    neighbors: tuple[tuple[int, ...], ...],
    *,
    min_component_edges: int,
    tau_high: float,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    if min_component_edges <= 1 or not np.any(mask):
        return mask.copy(), (), 0

    out = mask.copy()
    pruned_edges: list[int] = []
    pruned_components = 0
    for component in _component_members(mask, neighbors):
        if len(component) >= min_component_edges:
            continue
        if _component_has_cycle(component, mask, neighbors):
            continue
        component_idx = np.asarray(component, dtype=np.int64)
        if float(np.max(refined[component_idx])) >= tau_high:
            continue
        out[component_idx] = False
        pruned_edges.extend(int(edge_idx) for edge_idx in component)
        pruned_components += 1
    return out, tuple(sorted(pruned_edges)), pruned_components


def _component_vertex_degrees(component: tuple[int, ...], graph: _GraphViews) -> dict[int, int]:
    degrees: dict[int, int] = {}
    for edge_idx in component:
        vi, vj = graph.edge_to_vertices[int(edge_idx)]
        degrees[int(vi)] = degrees.get(int(vi), 0) + 1
        degrees[int(vj)] = degrees.get(int(vj), 0) + 1
    return degrees


def _component_has_mesh_cycle(component: tuple[int, ...], graph: _GraphViews) -> bool:
    vertex_count = len(_component_vertex_degrees(component, graph))
    return len(component) >= vertex_count if vertex_count else False


def _is_simple_open_path(component: tuple[int, ...], graph: _GraphViews) -> bool:
    if _component_has_mesh_cycle(component, graph):
        return False
    degrees = _component_vertex_degrees(component, graph)
    return all(degree <= 2 for degree in degrees.values())


def _trace_spur_chain(
    endpoint_vertex: int,
    mask: np.ndarray,
    graph: _GraphViews,
    seam_vertex_degrees: np.ndarray,
    *,
    max_spur_edges: int,
) -> tuple[tuple[int, ...], bool]:
    incident = [edge_idx for edge_idx in graph.vertex_to_edges[int(endpoint_vertex)] if mask[edge_idx]]
    if len(incident) != 1:
        return (), False

    chain: list[int] = []
    previous_vertex = int(endpoint_vertex)
    current_edge = int(incident[0])
    while True:
        if len(chain) >= max_spur_edges:
            return tuple(chain), True
        chain.append(current_edge)

        vi, vj = graph.edge_to_vertices[current_edge]
        next_vertex = int(vj) if int(vi) == previous_vertex else int(vi)
        if int(seam_vertex_degrees[next_vertex]) != 2:
            return tuple(chain), False

        next_edges = [
            int(edge_idx)
            for edge_idx in graph.vertex_to_edges[next_vertex]
            if mask[edge_idx] and int(edge_idx) != current_edge
        ]
        if len(next_edges) != 1:
            return tuple(chain), False
        previous_vertex, current_edge = next_vertex, next_edges[0]


def _seam_vertex_degrees(mask: np.ndarray, graph: _GraphViews) -> np.ndarray:
    degrees = np.zeros(graph.vertex_count, dtype=np.int64)
    for edge_idx in np.flatnonzero(mask):
        vi, vj = graph.edge_to_vertices[int(edge_idx)]
        degrees[int(vi)] += 1
        degrees[int(vj)] += 1
    return degrees


def _prune_spurs_once(
    mask: np.ndarray,
    refined: np.ndarray,
    neighbors: tuple[tuple[int, ...], ...],
    graph: _GraphViews,
    *,
    max_spur_edges: int,
    spur_mean_max: float,
    tau_high: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if max_spur_edges < 1 or not np.any(mask):
        return mask.copy(), ()

    to_remove: set[int] = set()
    seam_vertex_degrees = _seam_vertex_degrees(mask, graph)
    for component in _component_members(mask, neighbors):
        if _is_simple_open_path(component, graph):
            continue
        component_vertices = _component_vertex_degrees(component, graph)
        endpoint_vertices = sorted(vertex for vertex, degree in component_vertices.items() if degree == 1)
        for endpoint_vertex in endpoint_vertices:
            chain, over_cap = _trace_spur_chain(
                endpoint_vertex,
                mask,
                graph,
                seam_vertex_degrees,
                max_spur_edges=max_spur_edges,
            )
            if not chain or over_cap or len(chain) > max_spur_edges:
                continue
            if any(edge_idx in to_remove for edge_idx in chain):
                continue
            chain_idx = np.asarray(chain, dtype=np.int64)
            if float(np.mean(refined[chain_idx])) >= spur_mean_max:
                continue
            if float(np.max(refined[chain_idx])) >= tau_high:
                continue
            to_remove.update(int(edge_idx) for edge_idx in chain)

    if not to_remove:
        return mask.copy(), ()
    out = mask.copy()
    removed = tuple(sorted(to_remove))
    out[np.asarray(removed, dtype=np.int64)] = False
    return out, removed


def _prune_spurs(
    mask: np.ndarray,
    refined: np.ndarray,
    neighbors: tuple[tuple[int, ...], ...],
    graph: _GraphViews,
    *,
    max_spur_edges: int,
    spur_mean_max: float,
    tau_high: float,
    iteration_cap: int = 8,
) -> tuple[np.ndarray, tuple[int, ...]]:
    out = mask.copy()
    removed_all: list[int] = []
    for _ in range(iteration_cap):
        out, removed = _prune_spurs_once(
            out,
            refined,
            neighbors,
            graph,
            max_spur_edges=max_spur_edges,
            spur_mean_max=spur_mean_max,
            tau_high=tau_high,
        )
        if not removed:
            break
        removed_all.extend(removed)
    return out, tuple(sorted(set(removed_all)))


def _endpoint_vertices(mask: np.ndarray, graph: _GraphViews) -> tuple[int, ...]:
    if not np.any(mask):
        return ()
    seam_degree = np.zeros(graph.vertex_count, dtype=np.int64)
    for edge_idx in np.flatnonzero(mask):
        vi, vj = graph.edge_to_vertices[int(edge_idx)]
        seam_degree[int(vi)] += 1
        seam_degree[int(vj)] += 1
    return tuple(int(vertex) for vertex in np.flatnonzero(seam_degree == 1))


def _nonseam_vertex_adjacency(mask: np.ndarray, graph: _GraphViews, refined: np.ndarray, bridge_lambda: float) -> tuple[tuple[tuple[int, int, float, float], ...], ...]:
    adjacency: list[list[tuple[int, int, float, float]]] = [[] for _ in range(graph.vertex_count)]
    for edge_idx, (vi, vj) in enumerate(graph.edge_to_vertices):
        if mask[edge_idx]:
            continue
        length = float(graph.edge_lengths[edge_idx])
        cost = length * (1.0 + bridge_lambda * (1.0 - float(refined[edge_idx])))
        a = int(vi)
        b = int(vj)
        adjacency[a].append((b, int(edge_idx), cost, length))
        adjacency[b].append((a, int(edge_idx), cost, length))
    return tuple(tuple(sorted(values, key=lambda item: (item[0], item[1]))) for values in adjacency)


def _path_candidates_from_endpoint(
    source: int,
    endpoints: set[int],
    adjacency: tuple[tuple[tuple[int, int, float, float], ...], ...],
    refined: np.ndarray,
    *,
    max_bridge_edges: int,
    bridge_min_mean_conf: float,
    bridge_max_length: float,
    cost_cutoff: float,
) -> list[_BridgeCandidate]:
    candidates: list[_BridgeCandidate] = []
    heap: list[tuple[float, int, int, tuple[int, ...], tuple[int, ...], float]] = []
    heappush(heap, (0.0, 0, int(source), (int(source),), (), 0.0))

    while heap:
        total_cost, hops, vertex, vertices_path, edges_path, total_length = heappop(heap)
        if hops > max_bridge_edges or total_cost > cost_cutoff or total_length > bridge_max_length:
            continue
        if hops > 0 and vertex in endpoints and vertex > source:
            edge_idx = np.asarray(edges_path, dtype=np.int64)
            mean_conf = float(np.mean(refined[edge_idx])) if len(edge_idx) else 0.0
            if mean_conf >= bridge_min_mean_conf and total_length <= bridge_max_length:
                candidates.append(_BridgeCandidate(
                    endpoints=(int(source), int(vertex)),
                    edge_indices=tuple(int(idx) for idx in edges_path),
                    total_cost=float(total_cost),
                    total_length=float(total_length),
                    mean_confidence=mean_conf,
                ))
        if hops == max_bridge_edges:
            continue
        for next_vertex, edge_idx, edge_cost, edge_length in adjacency[vertex]:
            if next_vertex in vertices_path:
                continue
            next_cost = total_cost + edge_cost
            next_length = total_length + edge_length
            if next_cost > cost_cutoff or next_length > bridge_max_length:
                continue
            heappush(
                heap,
                (
                    next_cost,
                    hops + 1,
                    int(next_vertex),
                    vertices_path + (int(next_vertex),),
                    edges_path + (int(edge_idx),),
                    next_length,
                ),
            )
    return candidates


def _bridge_sort_key(candidate: _BridgeCandidate) -> tuple[int, float, tuple[int, int], tuple[int, ...]]:
    return (
        len(candidate.edge_indices),
        round(candidate.total_cost, 12),
        candidate.endpoints,
        candidate.edge_indices,
    )


def _local_bridges(
    mask: np.ndarray,
    refined: np.ndarray,
    graph: _GraphViews,
    *,
    max_bridge_edges: int,
    bridge_min_mean_conf: float,
    bridge_max_length: float,
    bridge_lambda: float,
) -> tuple[np.ndarray, tuple[int, ...], int, int]:
    endpoints = _endpoint_vertices(mask, graph)
    if len(endpoints) < 2 or max_bridge_edges < 1:
        return mask.copy(), (), len(endpoints), 0

    adjacency = _nonseam_vertex_adjacency(mask, graph, refined, bridge_lambda)
    endpoint_set = set(endpoints)
    cost_cutoff = bridge_max_length * (1.0 + max(0.0, bridge_lambda))

    best_by_pair: dict[tuple[int, int], _BridgeCandidate] = {}
    for source in endpoints:
        for candidate in _path_candidates_from_endpoint(
            int(source),
            endpoint_set,
            adjacency,
            refined,
            max_bridge_edges=max_bridge_edges,
            bridge_min_mean_conf=bridge_min_mean_conf,
            bridge_max_length=bridge_max_length,
            cost_cutoff=cost_cutoff,
        ):
            current = best_by_pair.get(candidate.endpoints)
            if current is None or _bridge_sort_key(candidate) < _bridge_sort_key(current):
                best_by_pair[candidate.endpoints] = candidate

    accepted_endpoints: set[int] = set()
    added_edges: set[int] = set()
    bridge_count = 0
    for candidate in sorted(best_by_pair.values(), key=_bridge_sort_key):
        if candidate.endpoints[0] in accepted_endpoints or candidate.endpoints[1] in accepted_endpoints:
            continue
        accepted_endpoints.update(candidate.endpoints)
        added_edges.update(candidate.edge_indices)
        bridge_count += 1

    out = mask.copy()
    if added_edges:
        out[np.asarray(sorted(added_edges), dtype=np.int64)] = True
    return out, tuple(sorted(added_edges)), len(endpoints), bridge_count


def _median_edge_length(edge_lengths: np.ndarray) -> float:
    if len(edge_lengths) == 0:
        return 0.0
    finite = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)]
    if len(finite) == 0:
        return 1.0
    return float(np.median(finite))


def apply_seam_postprocessing_detailed(
    topology: Any,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.60,
    max_gap_length: int = 3,
    min_island_size: int = 3,
    *,
    smoothing_beta: float = 0.30,
    smoothing_iterations: int = 1,
    tau_high: float | None = None,
    tau_low: float = 0.40,
    min_component_edges: int | None = None,
    max_spur_edges: int = 2,
    spur_mean_max: float = 0.50,
    bridge_lambda: float = 1.0,
    max_bridge_edges: int | None = None,
    bridge_min_mean_conf: float = 0.45,
    bridge_max_length: float | None = None,
    skeleton_threshold: float | None = None,
    skeleton_radius: int | None = None,
    distortion_per_edge: np.ndarray | None = None,
    confidence_weight: float = 1.0,
    distortion_weight: float = 1.0,
) -> SeamPostprocessResult:
    del skeleton_threshold, skeleton_radius, distortion_per_edge, confidence_weight, distortion_weight

    tau_high_value = float(threshold if tau_high is None else tau_high)
    tau_low_value = float(tau_low)
    _validate_probability_threshold('threshold', float(threshold))
    _validate_probability_threshold('tau_high', tau_high_value)
    _validate_probability_threshold('tau_low', tau_low_value)
    if tau_low_value > tau_high_value:
        raise ValueError(f'tau_low must be <= tau_high, got {tau_low_value} > {tau_high_value}')
    if smoothing_beta < 0.0 or smoothing_beta > 1.0:
        raise ValueError(f'smoothing_beta must be in [0, 1], got {smoothing_beta}')
    if smoothing_iterations < 0 or smoothing_iterations > 2:
        raise ValueError(f'smoothing_iterations must be between 0 and 2, got {smoothing_iterations}')
    if max_gap_length < 0:
        raise ValueError(f'max_gap_length must be non-negative, got {max_gap_length}')
    if min_island_size < 1:
        raise ValueError(f'min_island_size must be at least 1, got {min_island_size}')
    min_component_edges_value = int(min_island_size if min_component_edges is None else min_component_edges)
    max_bridge_edges_value = int(max_gap_length if max_bridge_edges is None else max_bridge_edges)
    if min_component_edges_value < 1:
        raise ValueError(f'min_component_edges must be at least 1, got {min_component_edges_value}')
    if max_spur_edges < 1:
        raise ValueError(f'max_spur_edges must be at least 1, got {max_spur_edges}')
    if spur_mean_max < 0.0 or spur_mean_max > 1.0:
        raise ValueError(f'spur_mean_max must be in [0, 1], got {spur_mean_max}')
    if max_bridge_edges_value < 0:
        raise ValueError(f'max_bridge_edges must be non-negative, got {max_bridge_edges_value}')
    if bridge_lambda < 0.0:
        raise ValueError(f'bridge_lambda must be non-negative, got {bridge_lambda}')
    if bridge_min_mean_conf < 0.0 or bridge_min_mean_conf > 1.0:
        raise ValueError(f'bridge_min_mean_conf must be in [0, 1], got {bridge_min_mean_conf}')

    probs = _as_probability_array(probabilities)
    edges = _as_unique_edges(unique_edges)
    if len(probs) != len(edges):
        raise ValueError(
            f'probabilities length {len(probs)} does not match unique_edges length {len(edges)}'
        )
    _validate_topology(topology, len(edges))

    if len(edges) == 0:
        empty = np.zeros(0, dtype=bool)
        return SeamPostprocessResult(
            threshold_mask=empty.copy(),
            skeleton_mask=empty.copy(),
            steiner_mask=empty.copy(),
            final_mask=empty.copy(),
            skeleton_deleted_vertices=(),
            steiner_added_edges=(),
            pruned_edge_indices=(),
            skeleton_terminal_vertex_count=0,
            steiner_terminal_group_count=0,
            steiner_tree_count=0,
            pruned_component_count=0,
        )

    graph = _build_graph_views(topology, edges)
    refined = _smooth_probabilities(
        probs,
        graph.dual_neighbors,
        beta=float(smoothing_beta),
        iterations=int(smoothing_iterations),
    )

    threshold_mask = refined >= tau_high_value
    hysteresis_mask = _hysteresis_threshold(refined, graph.seam_neighbors, tau_low_value, tau_high_value)
    current, tiny_pruned, tiny_component_count = _prune_tiny_components(
        hysteresis_mask,
        refined,
        graph.seam_neighbors,
        min_component_edges=min_component_edges_value,
        tau_high=tau_high_value,
    )
    current, spur_pruned = _prune_spurs(
        current,
        refined,
        graph.seam_neighbors,
        graph,
        max_spur_edges=int(max_spur_edges),
        spur_mean_max=float(spur_mean_max),
        tau_high=tau_high_value,
    )
    local_clean_mask = current.copy()

    bridge_length_limit = (
        2.5 * _median_edge_length(graph.edge_lengths)
        if bridge_max_length is None
        else float(bridge_max_length)
    )
    if bridge_length_limit < 0.0:
        raise ValueError(f'bridge_max_length must be non-negative, got {bridge_length_limit}')

    bridged, bridge_edges, endpoint_count, bridge_count = _local_bridges(
        current,
        refined,
        graph,
        max_bridge_edges=max_bridge_edges_value,
        bridge_min_mean_conf=float(bridge_min_mean_conf),
        bridge_max_length=bridge_length_limit,
        bridge_lambda=float(bridge_lambda),
    )

    final_mask, final_tiny_pruned, final_tiny_component_count = _prune_tiny_components(
        bridged,
        refined,
        graph.seam_neighbors,
        min_component_edges=min_component_edges_value,
        tau_high=tau_high_value,
    )
    final_mask, final_spur_pruned = _prune_spurs(
        final_mask,
        refined,
        graph.seam_neighbors,
        graph,
        max_spur_edges=int(max_spur_edges),
        spur_mean_max=float(spur_mean_max),
        tau_high=tau_high_value,
    )

    bridge_mask = np.zeros(len(edges), dtype=bool)
    if bridge_edges:
        bridge_mask[np.asarray(bridge_edges, dtype=np.int64)] = True
    pruned_edges = tuple(sorted(set(tiny_pruned + spur_pruned + final_tiny_pruned + final_spur_pruned)))

    return SeamPostprocessResult(
        threshold_mask=threshold_mask,
        skeleton_mask=local_clean_mask,
        steiner_mask=bridge_mask,
        final_mask=final_mask,
        skeleton_deleted_vertices=(),
        steiner_added_edges=bridge_edges,
        pruned_edge_indices=pruned_edges,
        skeleton_terminal_vertex_count=int(endpoint_count),
        steiner_terminal_group_count=int(len(_component_members(local_clean_mask, graph.seam_neighbors))),
        steiner_tree_count=int(bridge_count),
        pruned_component_count=int(tiny_component_count + final_tiny_component_count),
    )


def apply_seam_postprocessing(
    topology: Any,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.60,
    max_gap_length: int = 3,
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
    if len(seam_mask) != len(probabilities):
        raise ValueError('seam_mask must match probabilities shape')
    probabilities[seam_mask] = 1.0
    return apply_seam_postprocessing(
        topology=None,
        unique_edges=unique_edges,
        probabilities=probabilities,
        threshold=0.60,
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
