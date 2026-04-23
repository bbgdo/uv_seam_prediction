from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any

import networkx as nx
import numpy as np


_PROB_EPS = 1e-6


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
    vertex_graph: nx.Graph
    edge_neighbors: tuple[tuple[int, ...], ...]
    vertex_to_edges: tuple[tuple[int, ...], ...]
    edge_to_vertices: np.ndarray
    edge_lookup: dict[tuple[int, int], int]
    edge_lengths: np.ndarray
    vertex_count: int
    edge_count: int


@dataclass(frozen=True)
class _SeamComponent:
    component_id: int
    edge_indices: tuple[int, ...]
    vertex_indices: tuple[int, ...]
    seam_mass: float
    edge_count: int
    vertex_count: int
    endpoint_vertices: tuple[int, ...]
    junction_vertices: tuple[int, ...]
    cycle_rank: int

    @property
    def is_closed(self) -> bool:
        return len(self.endpoint_vertices) == 0 and self.cycle_rank >= 1

    @property
    def is_open(self) -> bool:
        return not self.is_closed

    @property
    def is_open_arc(self) -> bool:
        return (
            self.cycle_rank == 0
            and len(self.endpoint_vertices) == 2
            and len(self.junction_vertices) == 0
        )


@dataclass(frozen=True)
class _PathCandidate:
    source_vertex: int
    target_vertex: int
    edge_indices: tuple[int, ...]
    total_cost: float
    total_edges: int
    normalized_cost: float


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
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f'{name} must be a finite value in [0, 1], got {value}')


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
    if len(unique_edges) == 0:
        return (), {}, 0
    if np.any(unique_edges < 0):
        first = unique_edges[np.any(unique_edges < 0, axis=1)][0]
        raise ValueError(f'edge vertex ids must be non-negative, got {(int(first[0]), int(first[1]))}')

    vertex_count = int(unique_edges.max()) + 1
    vertex_to_edges_lists: list[list[int]] = [[] for _ in range(vertex_count)]
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_idx, (vi, vj) in enumerate(unique_edges):
        a = int(vi)
        b = int(vj)
        if a == b:
            raise ValueError(f'degenerate edge at index {edge_idx}: {(a, b)}')
        key = _canonical_edge_key(a, b)
        if key in edge_lookup:
            raise ValueError(f'duplicate edge {key}')
        edge_lookup[key] = int(edge_idx)
        vertex_to_edges_lists[a].append(int(edge_idx))
        vertex_to_edges_lists[b].append(int(edge_idx))
    vertex_to_edges = tuple(tuple(sorted(items)) for items in vertex_to_edges_lists)
    return vertex_to_edges, edge_lookup, vertex_count


def _build_edge_neighbors(vertex_to_edges: tuple[tuple[int, ...], ...], edge_count: int) -> tuple[tuple[int, ...], ...]:
    neighbors = [set() for _ in range(edge_count)]
    for incident in vertex_to_edges:
        for i, edge_a in enumerate(incident):
            for edge_b in incident[i + 1:]:
                neighbors[edge_a].add(edge_b)
                neighbors[edge_b].add(edge_a)
    return tuple(tuple(sorted(values)) for values in neighbors)


def _vertex_coordinates(topology: Any, vertex_count: int) -> np.ndarray | None:
    vertices = getattr(topology, 'canonical_vertices', None) if topology is not None else None
    if vertices is None:
        return None
    coords = np.asarray(vertices, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < vertex_count:
        return None
    return coords


def _edge_lengths(unique_edges: np.ndarray, coords: np.ndarray | None) -> np.ndarray:
    if len(unique_edges) == 0:
        return np.zeros(0, dtype=np.float64)
    if coords is None:
        return np.ones(len(unique_edges), dtype=np.float64)
    deltas = coords[unique_edges[:, 0]] - coords[unique_edges[:, 1]]
    lengths = np.linalg.norm(deltas, axis=1).astype(np.float64, copy=False)
    return np.where(lengths > 0.0, lengths, 1.0)


def _build_vertex_graph(unique_edges: np.ndarray, probabilities: np.ndarray, edge_lengths: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    for edge_idx, (vi, vj) in enumerate(unique_edges):
        graph.add_edge(
            int(vi),
            int(vj),
            edge_index=int(edge_idx),
            prob=float(probabilities[edge_idx]),
            edge_length=float(edge_lengths[edge_idx]),
        )
    return graph


def _build_graph_views(topology: Any, unique_edges: np.ndarray, probabilities: np.ndarray) -> _GraphViews:
    vertex_to_edges, edge_lookup, vertex_count = _build_incidence(unique_edges)
    coords = _vertex_coordinates(topology, vertex_count)
    edge_lengths = _edge_lengths(unique_edges, coords)
    return _GraphViews(
        vertex_graph=_build_vertex_graph(unique_edges, probabilities, edge_lengths),
        edge_neighbors=_build_edge_neighbors(vertex_to_edges, len(unique_edges)),
        vertex_to_edges=vertex_to_edges,
        edge_to_vertices=unique_edges.astype(np.int64, copy=True),
        edge_lookup=edge_lookup,
        edge_lengths=edge_lengths,
        vertex_count=vertex_count,
        edge_count=len(unique_edges),
    )


def _edge_costs(probabilities: np.ndarray, seam_threshold: float, lambda_off: float) -> np.ndarray:
    return -np.log(np.clip(probabilities, _PROB_EPS, 1.0)) + lambda_off * (probabilities < seam_threshold)


def _component_members(mask: np.ndarray, edge_neighbors: tuple[tuple[int, ...], ...]) -> list[tuple[int, ...]]:
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
            for neighbor in edge_neighbors[current]:
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(int(neighbor))
        components.append(tuple(sorted(members)))
    return components


def _component_vertices(edge_indices: tuple[int, ...], graph: _GraphViews) -> tuple[int, ...]:
    vertices: set[int] = set()
    for edge_idx in edge_indices:
        vi, vj = graph.edge_to_vertices[int(edge_idx)]
        vertices.add(int(vi))
        vertices.add(int(vj))
    return tuple(sorted(vertices))


def _component_vertex_degrees(edge_indices: tuple[int, ...], graph: _GraphViews) -> dict[int, int]:
    degrees: dict[int, int] = {}
    for edge_idx in edge_indices:
        vi, vj = graph.edge_to_vertices[int(edge_idx)]
        degrees[int(vi)] = degrees.get(int(vi), 0) + 1
        degrees[int(vj)] = degrees.get(int(vj), 0) + 1
    return degrees


def _analyze_components(mask: np.ndarray, graph: _GraphViews, probabilities: np.ndarray) -> list[_SeamComponent]:
    components: list[_SeamComponent] = []
    for component_id, edge_indices in enumerate(_component_members(mask, graph.edge_neighbors)):
        vertices = _component_vertices(edge_indices, graph)
        vertex_degrees = _component_vertex_degrees(edge_indices, graph)
        endpoint_vertices = tuple(sorted(vertex for vertex, degree in vertex_degrees.items() if degree == 1))
        junction_vertices = tuple(sorted(vertex for vertex, degree in vertex_degrees.items() if degree > 2))
        edge_count = len(edge_indices)
        vertex_count = len(vertices)
        cycle_rank = edge_count - vertex_count + 1
        seam_mass = float(np.sum(probabilities[np.asarray(edge_indices, dtype=np.int64)]))
        components.append(_SeamComponent(
            component_id=int(component_id),
            edge_indices=tuple(int(edge_idx) for edge_idx in edge_indices),
            vertex_indices=vertices,
            seam_mass=seam_mass,
            edge_count=edge_count,
            vertex_count=vertex_count,
            endpoint_vertices=endpoint_vertices,
            junction_vertices=junction_vertices,
            cycle_rank=int(cycle_rank),
        ))
    return components


def _choose_main_open_component(components: list[_SeamComponent]) -> _SeamComponent | None:
    open_components = [component for component in components if component.is_open]
    if not open_components:
        return None
    return min(
        open_components,
        key=lambda component: (-component.seam_mass, -component.edge_count, component.component_id),
    )


def _boundary_vertices(component: _SeamComponent, allow_all_if_no_endpoints: bool) -> tuple[int, ...]:
    if component.endpoint_vertices:
        return component.endpoint_vertices
    if allow_all_if_no_endpoints:
        return component.vertex_indices
    return ()


def _reconstruct_path(parent: dict[tuple[int, int], tuple[tuple[int, int], int]], end_state: tuple[int, int]) -> tuple[int, ...]:
    edge_indices: list[int] = []
    current = end_state
    while current in parent:
        previous, edge_idx = parent[current]
        edge_indices.append(int(edge_idx))
        current = previous
    edge_indices.reverse()
    return tuple(edge_indices)


def _bounded_shortest_path(
    graph: _GraphViews,
    edge_costs: np.ndarray,
    source: int,
    targets: set[int],
    *,
    max_edges: int,
    blocked_edges: set[int] | None = None,
    allowed_edges: set[int] | None = None,
    seam_mask: np.ndarray | None = None,
) -> list[_PathCandidate]:
    if source in targets:
        return []

    blocked_edges = blocked_edges or set()
    best_cost: dict[tuple[int, int], float] = {(int(source), 0): 0.0}
    parent: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    heap: list[tuple[float, int, int]] = [(0.0, 0, int(source))]
    candidates: list[_PathCandidate] = []

    while heap:
        total_cost, hops, vertex = heappop(heap)
        state = (int(vertex), int(hops))
        if total_cost > best_cost.get(state, np.inf):
            continue
        if hops > 0 and vertex in targets:
            edge_indices = _reconstruct_path(parent, state)
            candidates.append(_PathCandidate(
                source_vertex=int(source),
                target_vertex=int(vertex),
                edge_indices=edge_indices,
                total_cost=float(total_cost),
                total_edges=len(edge_indices),
                normalized_cost=float(total_cost / max(len(edge_indices), 1)),
            ))
            continue
        if hops >= max_edges:
            continue

        for neighbor in sorted(graph.vertex_graph.neighbors(vertex)):
            edge_index = int(graph.vertex_graph[vertex][neighbor]['edge_index'])
            if edge_index in blocked_edges:
                continue
            if allowed_edges is not None and edge_index not in allowed_edges:
                continue
            if seam_mask is not None and seam_mask[edge_index]:
                continue
            next_state = (int(neighbor), hops + 1)
            next_cost = total_cost + float(edge_costs[edge_index])
            previous_best = best_cost.get(next_state)
            if previous_best is not None and next_cost >= previous_best:
                continue
            best_cost[next_state] = next_cost
            parent[next_state] = (state, edge_index)
            heappush(heap, (next_cost, hops + 1, int(neighbor)))

    candidates.sort(key=lambda item: (round(item.total_cost, 12), item.total_edges, item.target_vertex, item.edge_indices))
    return candidates


def _path_new_edges(path: _PathCandidate, seam_mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(edge_idx) for edge_idx in path.edge_indices if not seam_mask[int(edge_idx)])


def _component_for_edge(edge_index: int, components: list[_SeamComponent]) -> _SeamComponent | None:
    for component in components:
        if edge_index in component.edge_indices:
            return component
    return None


def _edge_distance_map(source_edges: tuple[int, ...], edge_neighbors: tuple[tuple[int, ...], ...]) -> np.ndarray:
    distances = np.full(len(edge_neighbors), -1, dtype=np.int64)
    queue = deque(int(edge_idx) for edge_idx in source_edges)
    for edge_idx in queue:
        distances[int(edge_idx)] = 0
    while queue:
        current = queue.popleft()
        for neighbor in edge_neighbors[current]:
            if distances[neighbor] >= 0:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(int(neighbor))
    return distances


def _component_distance_to_edges(component: _SeamComponent, distance_map: np.ndarray) -> int | None:
    values = [int(distance_map[int(edge_idx)]) for edge_idx in component.edge_indices if distance_map[int(edge_idx)] >= 0]
    if not values:
        return None
    return min(values)


def _band_costs(edge_costs: np.ndarray, dist_to_main: np.ndarray, eta_main: float) -> np.ndarray:
    penalties = np.where(dist_to_main >= 0, eta_main * dist_to_main.astype(np.float64), eta_main * (np.max(dist_to_main[dist_to_main >= 0]) + 1 if np.any(dist_to_main >= 0) else 1.0))
    return edge_costs + penalties


def _collect_band_edges(
    component: _SeamComponent,
    main_component: _SeamComponent,
    graph: _GraphViews,
    *,
    r_band: int,
    r_snap: int,
    attachment_edges: tuple[int, ...],
) -> tuple[int, ...]:
    main_distance = _edge_distance_map(main_component.edge_indices, graph.edge_neighbors)
    seeds = set(int(edge_idx) for edge_idx in component.edge_indices)
    seeds.update(int(edge_idx) for edge_idx in main_component.edge_indices if 0 <= main_distance[int(edge_idx)] <= r_snap)
    seeds.update(int(edge_idx) for edge_idx in attachment_edges)
    band_edges = set(seeds)
    queue = deque((edge_idx, 0) for edge_idx in sorted(seeds))
    seen = set(seeds)
    while queue:
        edge_idx, depth = queue.popleft()
        if depth >= r_band:
            continue
        for neighbor in graph.edge_neighbors[int(edge_idx)]:
            if neighbor in seen:
                continue
            seen.add(int(neighbor))
            band_edges.add(int(neighbor))
            queue.append((int(neighbor), depth + 1))
    return tuple(sorted(band_edges))


def _lowest_attachment_paths(
    component: _SeamComponent,
    main_component: _SeamComponent,
    graph: _GraphViews,
    edge_costs: np.ndarray,
    seam_mask: np.ndarray,
    *,
    r_cross: int,
) -> list[_PathCandidate]:
    candidates: list[_PathCandidate] = []
    target_vertices = set(int(vertex) for vertex in main_component.vertex_indices)
    blocked_edges = set(int(edge_idx) for edge_idx in component.edge_indices)
    for source_vertex in _boundary_vertices(component, allow_all_if_no_endpoints=True):
        candidates.extend(_bounded_shortest_path(
            graph,
            edge_costs,
            int(source_vertex),
            target_vertices,
            max_edges=int(r_cross),
            blocked_edges=blocked_edges,
            seam_mask=seam_mask,
        ))
    candidates.sort(key=lambda item: (round(item.total_cost, 12), item.total_edges, item.source_vertex, item.target_vertex, item.edge_indices))
    return candidates


def _apply_band_collapse(
    mask: np.ndarray,
    component: _SeamComponent,
    main_component: _SeamComponent,
    graph: _GraphViews,
    edge_costs: np.ndarray,
    preserved_loops: set[frozenset[int]],
    *,
    r_snap: int,
    snap_max_edges: int,
    r_band: int,
    eta_main: float,
    r_cross: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if component.edge_count > snap_max_edges:
        return mask.copy(), ()
    if frozenset(component.edge_indices) in preserved_loops:
        return mask.copy(), ()

    distance_map = _edge_distance_map(main_component.edge_indices, graph.edge_neighbors)
    distance_to_main = _component_distance_to_edges(component, distance_map)
    if distance_to_main is None or distance_to_main > r_snap:
        return mask.copy(), ()

    attachment_paths = _lowest_attachment_paths(component, main_component, graph, edge_costs, mask, r_cross=r_cross)
    distinct_targets: list[_PathCandidate] = []
    seen_targets: set[int] = set()
    for candidate in attachment_paths:
        if candidate.target_vertex in seen_targets:
            continue
        seen_targets.add(candidate.target_vertex)
        distinct_targets.append(candidate)
        if len(distinct_targets) == 2:
            break
    if len(distinct_targets) < 2:
        return mask.copy(), ()

    attachment_edges = distinct_targets[0].edge_indices + distinct_targets[1].edge_indices
    band_edges = set(_collect_band_edges(
        component,
        main_component,
        graph,
        r_band=r_band,
        r_snap=r_snap,
        attachment_edges=attachment_edges,
    ))
    band_costs = _band_costs(edge_costs, distance_map, eta_main)
    backbone_candidates = _bounded_shortest_path(
        graph,
        band_costs,
        distinct_targets[0].target_vertex,
        {distinct_targets[1].target_vertex},
        max_edges=max(len(band_edges), 1),
        allowed_edges=band_edges,
    )
    if not backbone_candidates:
        return mask.copy(), ()

    backbone_edges = set(int(edge_idx) for edge_idx in backbone_candidates[0].edge_indices)
    backbone_edges.update(int(edge_idx) for edge_idx in attachment_edges)

    out = mask.copy()
    removed: list[int] = []
    component_edge_set = set(int(edge_idx) for edge_idx in component.edge_indices)
    for edge_idx in sorted(component_edge_set):
        if edge_idx in backbone_edges:
            continue
        out[edge_idx] = False
        removed.append(int(edge_idx))
    for edge_idx in backbone_edges:
        out[int(edge_idx)] = True
    return out, tuple(sorted(removed))


def _safe_loop_signature(component: _SeamComponent) -> frozenset[int]:
    return frozenset(int(edge_idx) for edge_idx in component.edge_indices)


def apply_seam_postprocessing_detailed(
    topology: Any,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
    max_gap_length: int = 8,
    min_island_size: int = 3,
    *,
    seam_threshold: float | None = None,
    lambda_off: float = 0.75,
    r_self: int = 6,
    r_cross: int = 8,
    tau_path: float = 1.35,
    kappa_self: float = 1.5,
    attach_margin: float = 0.10,
    garbage_max_edges: int = 4,
    r_snap: int = 3,
    snap_max_edges: int = 12,
    r_band: int = 2,
    eta_main: float = 0.35,
    smoothing_beta: float = 0.0,
    smoothing_iterations: int = 0,
    tau_high: float | None = None,
    tau_low: float | None = None,
    min_component_edges: int | None = None,
    max_island_edges: int | None = None,
    island_attach_hops: int | None = None,
    keep_small_cycle_conf: float | None = None,
    max_spur_edges: int = 2,
    spur_mean_max: float = 0.50,
    bridge_lambda: float | None = None,
    max_bridge_edges: int | None = None,
    bridge_min_mean_conf: float | None = None,
    bridge_max_length_factor: float | None = None,
    bridge_max_length: float | None = None,
    bridge_turn_weight: float | None = None,
    endpoint_tangent_span: int | None = None,
    theta_align_deg: float | None = None,
    reciprocal_bridge_only: bool | None = None,
    bridge_corridor_radius_factor: float | None = None,
    skeleton_threshold: float | None = None,
    skeleton_radius: int | None = None,
    distortion_per_edge: np.ndarray | None = None,
    confidence_weight: float = 1.0,
    distortion_weight: float = 1.0,
) -> SeamPostprocessResult:
    del (
        smoothing_beta,
        smoothing_iterations,
        tau_high,
        tau_low,
        min_component_edges,
        max_island_edges,
        island_attach_hops,
        keep_small_cycle_conf,
        max_spur_edges,
        spur_mean_max,
        bridge_lambda,
        max_bridge_edges,
        bridge_min_mean_conf,
        bridge_max_length_factor,
        bridge_max_length,
        bridge_turn_weight,
        endpoint_tangent_span,
        theta_align_deg,
        reciprocal_bridge_only,
        bridge_corridor_radius_factor,
        skeleton_threshold,
        skeleton_radius,
        distortion_per_edge,
        confidence_weight,
        distortion_weight,
        min_island_size,
        max_gap_length,
    )

    seam_threshold_value = float(threshold if seam_threshold is None else seam_threshold)
    _validate_probability_threshold('threshold', seam_threshold_value)
    if lambda_off < 0.0:
        raise ValueError(f'lambda_off must be non-negative, got {lambda_off}')
    if r_self < 0 or r_cross < 0:
        raise ValueError(f'r_self and r_cross must be non-negative, got {r_self}, {r_cross}')
    if tau_path < 0.0:
        raise ValueError(f'tau_path must be non-negative, got {tau_path}')
    if kappa_self < 0.0:
        raise ValueError(f'kappa_self must be non-negative, got {kappa_self}')
    if attach_margin < 0.0:
        raise ValueError(f'attach_margin must be non-negative, got {attach_margin}')
    if garbage_max_edges < 0 or r_snap < 0 or snap_max_edges < 0 or r_band < 0:
        raise ValueError('graph-radius parameters must be non-negative')
    if eta_main < 0.0:
        raise ValueError(f'eta_main must be non-negative, got {eta_main}')

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

    graph = _build_graph_views(topology, edges, probs)
    edge_costs = _edge_costs(probs, seam_threshold_value, float(lambda_off))

    threshold_mask = probs >= seam_threshold_value
    current_mask = threshold_mask.copy()
    added_bridge_edges: set[int] = set()
    preserved_loops: set[frozenset[int]] = set()
    pruned_edges: set[int] = set()
    pruned_component_count = 0

    # Stage A/B
    initial_components = _analyze_components(current_mask, graph, probs)
    main_component = _choose_main_open_component(initial_components)
    main_component_id = None if main_component is None else int(main_component.component_id)
    for component in initial_components:
        if component.component_id == main_component_id or not component.is_open_arc:
            continue
        max_self_edges = min(int(r_self), max(int(np.ceil(kappa_self * component.edge_count)), 0))
        if max_self_edges <= 0:
            continue
        endpoints = tuple(component.endpoint_vertices)
        candidates = _bounded_shortest_path(
            graph,
            edge_costs,
            endpoints[0],
            {endpoints[1]},
            max_edges=max_self_edges,
            blocked_edges=set(int(edge_idx) for edge_idx in component.edge_indices),
            seam_mask=current_mask,
        )
        if not candidates:
            continue
        candidate = candidates[0]
        if candidate.total_edges > r_self or candidate.normalized_cost > tau_path:
            continue
        for edge_idx in candidate.edge_indices:
            current_mask[int(edge_idx)] = True
            added_bridge_edges.add(int(edge_idx))
        updated_components = _analyze_components(current_mask, graph, probs)
        for updated in updated_components:
            if set(component.edge_indices).issubset(set(updated.edge_indices)) and updated.is_closed:
                preserved_loops.add(_safe_loop_signature(updated))
                break

    stage_b_mask = current_mask.copy()

    # Stage C
    bridge_count = 0
    while True:
        components = _analyze_components(current_mask, graph, probs)
        main_component = _choose_main_open_component(components)
        if main_component is None:
            break
        component_candidates: list[tuple[float, int, int, tuple[int, ...], _PathCandidate]] = []
        target_vertices = set(int(vertex) for vertex in main_component.vertex_indices)
        for component in components:
            if component.component_id == main_component.component_id or not component.is_open:
                continue
            blocked_edges = set(int(edge_idx) for edge_idx in component.edge_indices)
            candidates: list[_PathCandidate] = []
            for source_vertex in _boundary_vertices(component, allow_all_if_no_endpoints=True):
                candidates.extend(_bounded_shortest_path(
                    graph,
                    edge_costs,
                    int(source_vertex),
                    target_vertices,
                    max_edges=int(r_cross),
                    blocked_edges=blocked_edges,
                    seam_mask=current_mask,
                ))
            if not candidates:
                continue
            candidates = [candidate for candidate in candidates if candidate.total_edges <= r_cross and candidate.normalized_cost <= tau_path]
            if not candidates:
                continue
            candidates.sort(key=lambda item: (round(item.total_cost, 12), item.total_edges, item.source_vertex, item.target_vertex, item.edge_indices))
            if len(candidates) > 1 and not (candidates[0].normalized_cost + attach_margin < candidates[1].normalized_cost):
                continue
            component_candidates.append((
                float(candidates[0].total_cost),
                int(candidates[0].total_edges),
                int(component.component_id),
                candidates[0].edge_indices,
                candidates[0],
            ))
        if not component_candidates:
            break
        component_candidates.sort(key=lambda item: (round(item[0], 12), item[1], item[2], item[3]))
        chosen = component_candidates[0][4]
        for edge_idx in chosen.edge_indices:
            current_mask[int(edge_idx)] = True
            added_bridge_edges.add(int(edge_idx))
        bridge_count += 1

    # Stage D
    components_after_cross = _analyze_components(current_mask, graph, probs)
    main_component = _choose_main_open_component(components_after_cross)
    final_main_id = None if main_component is None else int(main_component.component_id)
    for component in components_after_cross:
        if not component.is_open:
            continue
        if component.component_id == final_main_id:
            continue
        current_mask[np.asarray(component.edge_indices, dtype=np.int64)] = False
        pruned_edges.update(int(edge_idx) for edge_idx in component.edge_indices)
        pruned_component_count += 1

    # Stage E
    components_before_snap = _analyze_components(current_mask, graph, probs)
    main_component = _choose_main_open_component(components_before_snap)
    if main_component is not None:
        for component in components_before_snap:
            if component.component_id == main_component.component_id:
                continue
            current_mask, removed = _apply_band_collapse(
                current_mask,
                component,
                main_component,
                graph,
                edge_costs,
                preserved_loops,
                r_snap=int(r_snap),
                snap_max_edges=int(snap_max_edges),
                r_band=int(r_band),
                eta_main=float(eta_main),
                r_cross=int(r_cross),
            )
            if removed:
                pruned_edges.update(int(edge_idx) for edge_idx in removed)

    # Stage F
    final_components = _analyze_components(current_mask, graph, probs)
    main_component = _choose_main_open_component(final_components)
    final_main_id = None if main_component is None else int(main_component.component_id)
    for component in final_components:
        if component.is_open and component.component_id != final_main_id:
            current_mask[np.asarray(component.edge_indices, dtype=np.int64)] = False
            pruned_edges.update(int(edge_idx) for edge_idx in component.edge_indices)
            pruned_component_count += 1

    final_components = _analyze_components(current_mask, graph, probs)
    open_main = _choose_main_open_component(final_components)
    endpoint_count = 0 if open_main is None else len(open_main.endpoint_vertices)

    bridge_mask = np.zeros(len(edges), dtype=bool)
    if added_bridge_edges:
        bridge_mask[np.asarray(sorted(added_bridge_edges), dtype=np.int64)] = True

    return SeamPostprocessResult(
        threshold_mask=threshold_mask,
        skeleton_mask=stage_b_mask,
        steiner_mask=bridge_mask,
        final_mask=current_mask.copy(),
        skeleton_deleted_vertices=(),
        steiner_added_edges=tuple(sorted(added_bridge_edges)),
        pruned_edge_indices=tuple(sorted(pruned_edges)),
        skeleton_terminal_vertex_count=int(endpoint_count),
        steiner_terminal_group_count=int(len(final_components)),
        steiner_tree_count=int(bridge_count),
        pruned_component_count=int(pruned_component_count),
    )


def apply_seam_postprocessing(
    topology: Any,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
    max_gap_length: int = 8,
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
        max_gap_length=8,
        min_island_size=min_component_size,
    )


def stitch_seam_gaps(
    probs: np.ndarray,
    seam_mask: np.ndarray,
    unique_edges: np.ndarray,
    edge_to_faces: dict | None = None,
    max_gap: int = 8,
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
        threshold=0.50,
        max_gap_length=max_gap,
        min_island_size=1,
    )


def postprocess_seams(
    probs: np.ndarray,
    unique_edges: np.ndarray,
    edge_to_faces: dict | None = None,
    threshold: float = 0.5,
    min_component_size: int = 3,
    max_gap: int = 8,
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
