from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


_PROB_EPS = 1e-6
_LOGGER = logging.getLogger(__name__)


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
    mean_prob: float
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

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoint_vertices)

    @property
    def junction_count(self) -> int:
        return len(self.junction_vertices)


@dataclass(frozen=True)
class _PathCandidate:
    source_vertex: int
    target_vertex: int
    edge_indices: tuple[int, ...]
    total_cost: float
    total_edges: int
    normalized_cost: float


@dataclass(frozen=True)
class _BridgeCandidate:
    source_component_id: int
    target_component_id: int | None
    source_vertex: int
    target_vertex: int
    edge_indices: tuple[int, ...]
    new_edges: tuple[int, ...]
    path_key: tuple[int, ...]
    total_cost: float
    mean_bridge_conf: float
    low_conf_fraction: float
    accepted_via_force_close: bool


@dataclass
class _BridgeStageStats:
    candidate_pairs_considered: int = 0
    shortest_paths_found: int = 0
    rejected_no_new_edges: int = 0
    rejected_by_length: int = 0
    rejected_by_third_party_protected: int = 0
    rejected_by_mean_conf: int = 0
    rejected_by_low_conf_fraction: int = 0
    rejected_by_ambiguity: int = 0
    duplicate_paths_collapsed: int = 0
    rejected_force_close_empty: int = 0
    rejected_force_close_third_party_protected: int = 0
    accepted_bridges: int = 0
    accepted_via_force_close: int = 0
    total_new_edges_added: int = 0
    total_components_merged: int = 0


@dataclass
class _StageE0Stats:
    e0_bands_considered: int = 0
    e0_bands_collapsed: int = 0
    e0_edges_removed: int = 0
    e0_edges_kept: int = 0
    e0_components_changed: int = 0


@dataclass
class _SpurStageStats:
    spur_chains_considered: int = 0
    spur_chains_removed: int = 0
    spur_edges_removed: int = 0


@dataclass(frozen=True)
class _BridgeTerminalRecord:
    stage_name: str
    component_id: int
    vertex_indices: tuple[int, ...]


@dataclass(frozen=True)
class _RejectedBridgeRecord:
    stage_name: str
    source_component_id: int
    target_component_id: int | None
    source_vertex: int
    target_vertex: int
    new_edges: tuple[int, ...]
    total_cost: float
    mean_bridge_conf: float
    low_conf_fraction: float
    rejection_reason: str


@dataclass(frozen=True)
class _AcceptedBridgeRecord:
    stage_name: str
    source_component_id: int
    target_component_id: int | None
    source_vertex: int
    target_vertex: int
    new_edges: tuple[int, ...]


@dataclass(frozen=True)
class _E0BandRecord:
    component_id: int
    kept_edge_ids: tuple[int, ...]
    removed_edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class _RemovedSpurRecord:
    source_vertex: int
    attach_vertex: int
    chain_edges: tuple[int, ...]
    mean_conf: float
    added_fraction: float


@dataclass
class _BridgeDebugExport:
    export_dir: Path
    terminals: list[_BridgeTerminalRecord]
    rejected_bridges: list[_RejectedBridgeRecord]
    accepted_bridges: list[_AcceptedBridgeRecord]
    e0_bands: list[_E0BandRecord]
    removed_spurs: list[_RemovedSpurRecord]
    accepted_bridge_edge_order: list[int]
    removed_bridge_reasons: dict[int, str]
    persistence_checks: dict[str, dict[str, Any]]


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


def _compute_search_edge_costs(probabilities: np.ndarray, alpha_cost: float) -> np.ndarray:
    return 1.0 + alpha_cost * (1.0 - probabilities)


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
        mean_prob = float(seam_mass / max(edge_count, 1))
        components.append(_SeamComponent(
            component_id=int(component_id),
            edge_indices=tuple(int(edge_idx) for edge_idx in edge_indices),
            vertex_indices=vertices,
            seam_mass=seam_mass,
            mean_prob=mean_prob,
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


def _build_terminal_set(component: _SeamComponent, seam_vertex_degrees: dict[int, int]) -> tuple[int, ...]:
    terminals = tuple(sorted(vertex for vertex, degree in seam_vertex_degrees.items() if degree != 2))
    if terminals:
        return terminals
    return component.vertex_indices


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


def _path_new_edges_from_indices(path_edge_indices: tuple[int, ...], seam_mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(edge_idx) for edge_idx in path_edge_indices if not seam_mask[int(edge_idx)])


def _build_edge_component_ids(components: list[_SeamComponent], edge_count: int) -> np.ndarray:
    component_ids = np.full(edge_count, -1, dtype=np.int64)
    for component in components:
        component_ids[np.asarray(component.edge_indices, dtype=np.int64)] = int(component.component_id)
    return component_ids


def _path_uses_blocked_third_party_seam(
    path_edge_indices: tuple[int, ...],
    seam_mask: np.ndarray,
    edge_component_ids: np.ndarray,
    allowed_component_ids: set[int],
    blocked_component_ids: set[int],
) -> bool:
    for edge_idx in path_edge_indices:
        edge_idx = int(edge_idx)
        if not seam_mask[edge_idx]:
            continue
        component_id = int(edge_component_ids[edge_idx])
        if component_id >= 0 and component_id in blocked_component_ids and component_id not in allowed_component_ids:
            return True
    return False


def _candidate_path_key(new_edges: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(edge_idx) for edge_idx in new_edges))


def _candidate_edge_jaccard(a: _BridgeCandidate, b: _BridgeCandidate) -> float:
    set_a = set(int(edge_idx) for edge_idx in a.new_edges)
    set_b = set(int(edge_idx) for edge_idx in b.new_edges)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return float(len(set_a & set_b) / len(union))


def _bridge_candidate_sort_key(candidate: _BridgeCandidate) -> tuple[float, int, float, tuple[int, ...]]:
    return (
        -candidate.mean_bridge_conf,
        len(candidate.new_edges),
        round(candidate.total_cost, 12),
        (
            int(candidate.source_component_id),
            -1 if candidate.target_component_id is None else int(candidate.target_component_id),
            int(candidate.source_vertex),
            int(candidate.target_vertex),
            *candidate.path_key,
        ),
    )


def _rank_bridge_candidates(candidates: list[_BridgeCandidate]) -> list[_BridgeCandidate]:
    return sorted(candidates, key=_bridge_candidate_sort_key)


def _subgraph_vertex_degrees(edge_indices: tuple[int, ...], graph: _GraphViews) -> dict[int, int]:
    return _component_vertex_degrees(edge_indices, graph)


def _subgraph_cycle_rank(edge_indices: tuple[int, ...], graph: _GraphViews) -> int:
    vertex_count = len(_component_vertices(edge_indices, graph))
    return int(len(edge_indices) - vertex_count + 1)


def _seam_edge_neighborhood(
    seed_edge: int,
    component_edge_set: set[int],
    graph: _GraphViews,
    *,
    radius: int,
) -> tuple[int, ...]:
    band_edges = {int(seed_edge)}
    queue = deque([(int(seed_edge), 0)])
    seen = {int(seed_edge)}
    while queue:
        edge_idx, depth = queue.popleft()
        if depth >= radius:
            continue
        for neighbor in graph.edge_neighbors[int(edge_idx)]:
            neighbor = int(neighbor)
            if neighbor in seen or neighbor not in component_edge_set:
                continue
            seen.add(neighbor)
            band_edges.add(neighbor)
            queue.append((neighbor, depth + 1))
    return tuple(sorted(band_edges))


def _band_boundary_vertices(
    band_edges: tuple[int, ...],
    component_edge_set: set[int],
    graph: _GraphViews,
) -> tuple[int, ...]:
    band_set = set(int(edge_idx) for edge_idx in band_edges)
    band_degrees = _subgraph_vertex_degrees(band_edges, graph)
    boundary: set[int] = set()
    for vertex_idx in _component_vertices(band_edges, graph):
        incident_component_edges = [
            int(edge_idx)
            for edge_idx in graph.vertex_to_edges[int(vertex_idx)]
            if int(edge_idx) in component_edge_set
        ]
        local_degree = int(band_degrees.get(int(vertex_idx), 0))
        has_external_component_edge = any(edge_idx not in band_set for edge_idx in incident_component_edges)
        if has_external_component_edge or local_degree != 2:
            boundary.add(int(vertex_idx))
    if boundary:
        return tuple(sorted(boundary))
    irregular = tuple(sorted(vertex for vertex, degree in band_degrees.items() if degree != 2))
    if irregular:
        return irregular
    return ()


def _band_diameter_vertex_pairs(
    band_edges: tuple[int, ...],
    graph: _GraphViews,
) -> list[tuple[int, int]]:
    band_edge_set = set(int(edge_idx) for edge_idx in band_edges)
    band_vertices = _component_vertices(band_edges, graph)
    max_hops = -1
    pairs: list[tuple[int, int]] = []
    for index, source_vertex in enumerate(band_vertices):
        paths = _bounded_shortest_path(
            graph,
            np.ones(graph.edge_count, dtype=np.float64),
            int(source_vertex),
            set(int(vertex_idx) for vertex_idx in band_vertices[index + 1:]),
            max_edges=max(len(band_edges), 1),
            allowed_edges=band_edge_set,
        )
        for path in paths:
            hops = len(path.edge_indices)
            pair = tuple(sorted((int(path.source_vertex), int(path.target_vertex))))
            if hops > max_hops:
                max_hops = hops
                pairs = [pair]
            elif hops == max_hops:
                pairs.append(pair)
    return sorted(set(pairs))


def _band_path_score(
    edge_indices: tuple[int, ...],
    probabilities: np.ndarray,
    *,
    e0_length_penalty: float,
) -> float:
    if not edge_indices:
        return -np.inf
    probs = probabilities[np.asarray(edge_indices, dtype=np.int64)]
    return float(np.sum(probs) - e0_length_penalty * len(edge_indices))


def _best_band_representative_path(
    band_edges: tuple[int, ...],
    component_edge_set: set[int],
    graph: _GraphViews,
    probabilities: np.ndarray,
    *,
    e0_length_penalty: float,
) -> tuple[int, ...]:
    boundary_vertices = _band_boundary_vertices(band_edges, component_edge_set, graph)
    keep_costs = (1.0 + float(e0_length_penalty)) - probabilities
    best_score = -np.inf
    best_path: tuple[int, ...] = ()
    band_allowed_edges = set(int(edge_idx) for edge_idx in band_edges)
    max_edges = max(len(band_edges), 1)
    terminal_pairs: list[tuple[int, int]] = []
    if len(boundary_vertices) >= 2:
        for index, source_vertex in enumerate(boundary_vertices):
            for target_vertex in boundary_vertices[index + 1:]:
                terminal_pairs.append((int(source_vertex), int(target_vertex)))
    else:
        terminal_pairs.extend(_band_diameter_vertex_pairs(band_edges, graph))
    for source_vertex, target_vertex in terminal_pairs:
        paths = _bounded_shortest_path(
            graph,
            keep_costs,
            int(source_vertex),
            {int(target_vertex)},
            max_edges=max_edges,
            allowed_edges=band_allowed_edges,
        )
        if not paths:
            continue
        path_edges = tuple(int(edge_idx) for edge_idx in paths[0].edge_indices)
        if not path_edges:
            continue
        score = _band_path_score(path_edges, probabilities, e0_length_penalty=float(e0_length_penalty))
        candidate_key = tuple(int(edge_idx) for edge_idx in path_edges)
        best_key = tuple(int(edge_idx) for edge_idx in best_path)
        if (
            score > best_score
            or (
                np.isclose(score, best_score)
                and (
                    len(path_edges) < len(best_path)
                    or (len(path_edges) == len(best_path) and candidate_key < best_key)
                )
            )
        ):
            best_score = float(score)
            best_path = path_edges
    return best_path


def _collect_stage_e0_band_keys(
    component: _SeamComponent,
    graph: _GraphViews,
    *,
    e0_radius: int,
) -> list[tuple[int, ...]]:
    component_edge_set = set(int(edge_idx) for edge_idx in component.edge_indices)
    seen_band_keys: set[tuple[int, ...]] = set()
    band_keys: list[tuple[int, ...]] = []
    max_band_edges = max(6, 4 * int(e0_radius) + 4)
    for seed_edge in component.edge_indices:
        band_edges = _seam_edge_neighborhood(int(seed_edge), component_edge_set, graph, radius=int(e0_radius))
        if len(band_edges) < 4 or len(band_edges) > max_band_edges:
            continue
        band_degrees = _subgraph_vertex_degrees(band_edges, graph)
        if _subgraph_cycle_rank(band_edges, graph) < 1 and max(band_degrees.values(), default=0) <= 2:
            continue
        band_key = tuple(sorted(int(edge_idx) for edge_idx in band_edges))
        if band_key in seen_band_keys:
            continue
        seen_band_keys.add(band_key)
        band_keys.append(band_key)
    band_keys.sort(key=lambda item: (len(item), item))
    return band_keys


def _apply_stage_e0(
    seam_mask: np.ndarray,
    graph: _GraphViews,
    probabilities: np.ndarray,
    edge_origin: np.ndarray,
    *,
    e0_radius: int,
    e0_length_penalty: float,
    debug_export: _BridgeDebugExport | None,
) -> tuple[np.ndarray, _StageE0Stats]:
    stats = _StageE0Stats()
    if e0_radius <= 0:
        return seam_mask.copy(), stats

    working_mask = seam_mask.copy()
    used_edges: set[int] = set()
    components_before = _analyze_components(working_mask, graph, probabilities)
    changed_component_ids: set[int] = set()

    for component in components_before:
        component_edge_set = set(int(edge_idx) for edge_idx in component.edge_indices)
        for band_edges in _collect_stage_e0_band_keys(component, graph, e0_radius=int(e0_radius)):
            if any(int(edge_idx) in used_edges for edge_idx in band_edges):
                continue
            if any(not bool(working_mask[int(edge_idx)]) for edge_idx in band_edges):
                continue
            stats.e0_bands_considered += 1
            kept_path = _best_band_representative_path(
                band_edges,
                component_edge_set,
                graph,
                probabilities,
                e0_length_penalty=float(e0_length_penalty),
            )
            if not kept_path:
                continue
            kept_edges = tuple(sorted(int(edge_idx) for edge_idx in kept_path))
            removed_edges = tuple(
                int(edge_idx)
                for edge_idx in band_edges
                if int(edge_idx) not in set(int(kept) for kept in kept_edges)
            )
            if not removed_edges:
                continue
            for edge_idx in removed_edges:
                working_mask[int(edge_idx)] = False
                used_edges.add(int(edge_idx))
            for edge_idx in kept_edges:
                edge_origin[int(edge_idx)] = 'stage_e0'
                used_edges.add(int(edge_idx))
            stats.e0_bands_collapsed += 1
            stats.e0_edges_removed += len(removed_edges)
            stats.e0_edges_kept += len(kept_edges)
            changed_component_ids.add(int(component.component_id))
            _record_e0_band(debug_export, int(component.component_id), kept_edges, removed_edges)

    stats.e0_components_changed = len(changed_component_ids)
    return working_mask, stats


def _walk_spur_chain(
    start_vertex: int,
    component_edges: set[int],
    seam_vertex_degrees: dict[int, int],
    graph: _GraphViews,
    *,
    max_spur_edges: int,
) -> tuple[int, int, tuple[int, ...]] | None:
    current_vertex = int(start_vertex)
    previous_edge = -1
    chain_edges: list[int] = []
    while len(chain_edges) < max_spur_edges:
        available_edges = [
            int(edge_idx)
            for edge_idx in graph.vertex_to_edges[current_vertex]
            if int(edge_idx) in component_edges and int(edge_idx) != previous_edge
        ]
        if not available_edges:
            break
        if len(chain_edges) > 0 and seam_vertex_degrees.get(current_vertex, 0) != 2:
            break
        next_edge = min(available_edges)
        chain_edges.append(int(next_edge))
        vi, vj = graph.edge_to_vertices[int(next_edge)]
        next_vertex = int(vj) if int(vi) == current_vertex else int(vi)
        previous_edge = int(next_edge)
        current_vertex = next_vertex
        if seam_vertex_degrees.get(current_vertex, 0) != 2:
            break
    if not chain_edges:
        return None
    return int(start_vertex), int(current_vertex), tuple(chain_edges)


def _apply_spur_cleanup(
    seam_mask: np.ndarray,
    graph: _GraphViews,
    probabilities: np.ndarray,
    edge_origin: np.ndarray,
    *,
    max_spur_edges: int,
    spur_mean_conf: float,
    spur_added_fraction_min: float,
    debug_export: _BridgeDebugExport | None,
) -> tuple[np.ndarray, _SpurStageStats, tuple[int, ...]]:
    stats = _SpurStageStats()
    if max_spur_edges <= 0:
        return seam_mask.copy(), stats, ()

    working_mask = seam_mask.copy()
    removed_edges: set[int] = set()
    removable_origins = {'bridge_b', 'bridge_c', 'stage_e0', 'stage_e'}
    components = _analyze_components(working_mask, graph, probabilities)
    for component in components:
        seam_vertex_degrees = _component_vertex_degrees(component.edge_indices, graph)
        component_edges = set(int(edge_idx) for edge_idx in component.edge_indices)
        leaf_vertices = sorted(
            int(vertex_idx)
            for vertex_idx, degree in seam_vertex_degrees.items()
            if int(degree) == 1
        )
        for leaf_vertex in leaf_vertices:
            walked = _walk_spur_chain(
                int(leaf_vertex),
                component_edges,
                seam_vertex_degrees,
                graph,
                max_spur_edges=int(max_spur_edges),
            )
            if walked is None:
                continue
            source_vertex, attach_vertex, chain_edges = walked
            if any(int(edge_idx) in removed_edges for edge_idx in chain_edges):
                continue
            stats.spur_chains_considered += 1
            if len(chain_edges) > max_spur_edges:
                continue
            if seam_vertex_degrees.get(int(attach_vertex), 0) < 3:
                continue
            mean_conf = float(np.mean(probabilities[np.asarray(chain_edges, dtype=np.int64)]))
            added_fraction = float(np.mean([
                str(edge_origin[int(edge_idx)]) in removable_origins
                for edge_idx in chain_edges
            ]))
            if mean_conf >= spur_mean_conf or added_fraction < spur_added_fraction_min:
                continue
            for edge_idx in chain_edges:
                working_mask[int(edge_idx)] = False
                removed_edges.add(int(edge_idx))
            stats.spur_chains_removed += 1
            stats.spur_edges_removed += len(chain_edges)
            _record_removed_spur(
                debug_export,
                source_vertex=int(source_vertex),
                attach_vertex=int(attach_vertex),
                chain_edges=chain_edges,
                mean_conf=mean_conf,
                added_fraction=added_fraction,
            )

    return working_mask, stats, tuple(sorted(removed_edges))


def _log_bridge_stats(stage_name: str, stats: _BridgeStageStats) -> None:
    if not _LOGGER.isEnabledFor(logging.DEBUG):
        return
    _LOGGER.debug(
        (
            'Postprocess %s stats: candidates=%d paths=%d no_new=%d length=%d '
            'third_party_protected=%d mean_conf=%d low_conf=%d ambiguity=%d '
            'duplicates=%d accepted=%d force_close=%d new_edges=%d merged=%d'
        ),
        stage_name,
        int(stats.candidate_pairs_considered),
        int(stats.shortest_paths_found),
        int(stats.rejected_no_new_edges),
        int(stats.rejected_by_length),
        int(stats.rejected_by_third_party_protected),
        int(stats.rejected_by_mean_conf),
        int(stats.rejected_by_low_conf_fraction),
        int(stats.rejected_by_ambiguity),
        int(stats.duplicate_paths_collapsed),
        int(stats.accepted_bridges),
        int(stats.accepted_via_force_close),
        int(stats.total_new_edges_added),
        int(stats.total_components_merged),
    )


def _bridge_stage_stats_payload(stats: _BridgeStageStats) -> dict[str, int]:
    return {
        'candidate_pairs_considered': int(stats.candidate_pairs_considered),
        'shortest_paths_found': int(stats.shortest_paths_found),
        'rejected_no_new_edges': int(stats.rejected_no_new_edges),
        'rejected_by_length': int(stats.rejected_by_length),
        'rejected_by_third_party_protected': int(stats.rejected_by_third_party_protected),
        'rejected_by_mean_conf': int(stats.rejected_by_mean_conf),
        'rejected_by_low_conf_fraction': int(stats.rejected_by_low_conf_fraction),
        'rejected_by_ambiguity': int(stats.rejected_by_ambiguity),
        'duplicate_paths_collapsed': int(stats.duplicate_paths_collapsed),
        'rejected_force_close_empty': int(stats.rejected_force_close_empty),
        'rejected_force_close_third_party_protected': int(stats.rejected_force_close_third_party_protected),
        'accepted_bridges': int(stats.accepted_bridges),
        'accepted_via_force_close': int(stats.accepted_via_force_close),
        'total_new_edges_added': int(stats.total_new_edges_added),
        'total_components_merged': int(stats.total_components_merged),
    }


def _stage_e0_stats_payload(stats: _StageE0Stats) -> dict[str, int]:
    return {
        'e0_bands_considered': int(stats.e0_bands_considered),
        'e0_bands_collapsed': int(stats.e0_bands_collapsed),
        'e0_edges_removed': int(stats.e0_edges_removed),
        'e0_edges_kept': int(stats.e0_edges_kept),
        'e0_components_changed': int(stats.e0_components_changed),
    }


def _spur_stage_stats_payload(stats: _SpurStageStats) -> dict[str, int]:
    return {
        'spur_chains_considered': int(stats.spur_chains_considered),
        'spur_chains_removed': int(stats.spur_chains_removed),
        'spur_edges_removed': int(stats.spur_edges_removed),
    }


def _log_stage_e0_stats(stats: _StageE0Stats) -> None:
    if not _LOGGER.isEnabledFor(logging.DEBUG):
        return
    _LOGGER.debug(
        'Postprocess thickness_collapse stats: bands=%d collapsed=%d edges_removed=%d edges_kept=%d components_changed=%d',
        int(stats.e0_bands_considered),
        int(stats.e0_bands_collapsed),
        int(stats.e0_edges_removed),
        int(stats.e0_edges_kept),
        int(stats.e0_components_changed),
    )


def _log_spur_stats(stats: _SpurStageStats) -> None:
    if not _LOGGER.isEnabledFor(logging.DEBUG):
        return
    _LOGGER.debug(
        'Postprocess spur_cleanup stats: chains=%d removed=%d edges_removed=%d',
        int(stats.spur_chains_considered),
        int(stats.spur_chains_removed),
        int(stats.spur_edges_removed),
    )


def _make_debug_export(debug_export_dir: str | Path | None) -> _BridgeDebugExport | None:
    if debug_export_dir in (None, ''):
        return None
    export_dir = Path(debug_export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    return _BridgeDebugExport(
        export_dir=export_dir,
        terminals=[],
        rejected_bridges=[],
        accepted_bridges=[],
        e0_bands=[],
        removed_spurs=[],
        accepted_bridge_edge_order=[],
        removed_bridge_reasons={},
        persistence_checks={},
    )


def _record_terminals(
    debug_export: _BridgeDebugExport | None,
    stage_name: str,
    component_id: int,
    terminals: tuple[int, ...],
) -> None:
    if debug_export is None:
        return
    debug_export.terminals.append(_BridgeTerminalRecord(
        stage_name=str(stage_name),
        component_id=int(component_id),
        vertex_indices=tuple(int(vertex_idx) for vertex_idx in terminals),
    ))


def _record_rejected_bridge(
    debug_export: _BridgeDebugExport | None,
    stage_name: str,
    path: _PathCandidate,
    new_edges: tuple[int, ...],
    mean_bridge_conf: float,
    low_conf_fraction: float,
    rejection_reason: str,
    *,
    source_component_id: int,
    target_component_id: int | None,
) -> None:
    if debug_export is None:
        return
    debug_export.rejected_bridges.append(_RejectedBridgeRecord(
        stage_name=str(stage_name),
        source_component_id=int(source_component_id),
        target_component_id=None if target_component_id is None else int(target_component_id),
        source_vertex=int(path.source_vertex),
        target_vertex=int(path.target_vertex),
        new_edges=tuple(int(edge_idx) for edge_idx in new_edges),
        total_cost=float(path.total_cost),
        mean_bridge_conf=float(mean_bridge_conf),
        low_conf_fraction=float(low_conf_fraction),
        rejection_reason=str(rejection_reason),
    ))


def _record_accepted_bridge(
    debug_export: _BridgeDebugExport | None,
    stage_name: str,
    candidate: _BridgeCandidate,
) -> None:
    if debug_export is None:
        return
    debug_export.accepted_bridges.append(_AcceptedBridgeRecord(
        stage_name=str(stage_name),
        source_component_id=int(candidate.source_component_id),
        target_component_id=None if candidate.target_component_id is None else int(candidate.target_component_id),
        source_vertex=int(candidate.source_vertex),
        target_vertex=int(candidate.target_vertex),
        new_edges=tuple(int(edge_idx) for edge_idx in candidate.new_edges),
    ))


def _record_e0_band(
    debug_export: _BridgeDebugExport | None,
    component_id: int,
    kept_edge_ids: tuple[int, ...],
    removed_edge_ids: tuple[int, ...],
) -> None:
    if debug_export is None:
        return
    debug_export.e0_bands.append(_E0BandRecord(
        component_id=int(component_id),
        kept_edge_ids=tuple(int(edge_idx) for edge_idx in kept_edge_ids),
        removed_edge_ids=tuple(int(edge_idx) for edge_idx in removed_edge_ids),
    ))


def _record_removed_spur(
    debug_export: _BridgeDebugExport | None,
    *,
    source_vertex: int,
    attach_vertex: int,
    chain_edges: tuple[int, ...],
    mean_conf: float,
    added_fraction: float,
) -> None:
    if debug_export is None:
        return
    debug_export.removed_spurs.append(_RemovedSpurRecord(
        source_vertex=int(source_vertex),
        attach_vertex=int(attach_vertex),
        chain_edges=tuple(int(edge_idx) for edge_idx in chain_edges),
        mean_conf=float(mean_conf),
        added_fraction=float(added_fraction),
    ))


def _record_accepted_bridge_edges(
    debug_export: _BridgeDebugExport | None,
    new_edges: tuple[int, ...],
) -> None:
    if debug_export is None:
        return
    seen = set(int(edge_idx) for edge_idx in debug_export.accepted_bridge_edge_order)
    for edge_idx in new_edges:
        edge_idx = int(edge_idx)
        if edge_idx not in seen:
            debug_export.accepted_bridge_edge_order.append(edge_idx)
            seen.add(edge_idx)


def _mark_removed_bridge_edges(
    debug_export: _BridgeDebugExport | None,
    removed_edges: tuple[int, ...] | list[int] | np.ndarray,
    removal_reason: str,
) -> None:
    if debug_export is None:
        return
    accepted_edges = set(int(edge_idx) for edge_idx in debug_export.accepted_bridge_edge_order)
    for edge_idx in removed_edges:
        edge_idx = int(edge_idx)
        if edge_idx in accepted_edges and edge_idx not in debug_export.removed_bridge_reasons:
            debug_export.removed_bridge_reasons[edge_idx] = str(removal_reason)


def _record_bridge_persistence_checkpoint(
    debug_export: _BridgeDebugExport | None,
    checkpoint_name: str,
    working_seam_mask: np.ndarray,
    accepted_bridge_edges: set[int],
) -> None:
    if debug_export is None:
        return
    accepted_edges_sorted = tuple(sorted(int(edge_idx) for edge_idx in accepted_bridge_edges))
    present_count = sum(1 for edge_idx in accepted_edges_sorted if bool(working_seam_mask[int(edge_idx)]))
    missing_without_reason = sum(
        1
        for edge_idx in accepted_edges_sorted
        if not bool(working_seam_mask[int(edge_idx)])
        and int(edge_idx) not in debug_export.removed_bridge_reasons
    )
    debug_export.persistence_checks[str(checkpoint_name)] = {
        'accepted_bridge_edges_total': int(len(accepted_edges_sorted)),
        'accepted_bridge_edges_present': int(present_count),
        'accepted_bridge_edges_missing_without_reason': int(missing_without_reason),
        'removed_edges': {
            str(int(edge_idx)): debug_export.removed_bridge_reasons[int(edge_idx)]
            for edge_idx in accepted_edges_sorted
            if int(edge_idx) in debug_export.removed_bridge_reasons
        },
    }


def _debug_vertex_coords(graph: _GraphViews, topology: Any) -> np.ndarray:
    coords = _vertex_coordinates(topology, graph.vertex_count)
    if coords is not None:
        return coords
    fallback = np.zeros((graph.vertex_count, 3), dtype=np.float64)
    fallback[:, 0] = np.arange(graph.vertex_count, dtype=np.float64)
    return fallback


def _write_terminals_obj(
    path: Path,
    coords: np.ndarray,
    terminal_records: list[_BridgeTerminalRecord],
) -> None:
    lines = ['# Stage B/C terminal vertices']
    vertex_counter = 0
    for record in terminal_records:
        lines.append(f'o {record.stage_name}_component_{record.component_id}')
        point_indices: list[int] = []
        for vertex_idx in record.vertex_indices:
            x, y, z = coords[int(vertex_idx)]
            lines.append(f'v {x:.9g} {y:.9g} {z:.9g}')
            vertex_counter += 1
            point_indices.append(vertex_counter)
        if point_indices:
            lines.append('p ' + ' '.join(str(index) for index in point_indices))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _write_accepted_bridges_obj(
    path: Path,
    coords: np.ndarray,
    graph: _GraphViews,
    accepted_records: list[_AcceptedBridgeRecord],
) -> None:
    lines = ['# Accepted Stage B/C bridge edges']
    vertex_counter = 0
    for index, record in enumerate(accepted_records):
        lines.append(f'o {record.stage_name}_bridge_{index}')
        for edge_idx in record.new_edges:
            vi, vj = graph.edge_to_vertices[int(edge_idx)]
            ax, ay, az = coords[int(vi)]
            bx, by, bz = coords[int(vj)]
            lines.append(f'v {ax:.9g} {ay:.9g} {az:.9g}')
            lines.append(f'v {bx:.9g} {by:.9g} {bz:.9g}')
            lines.append(f'l {vertex_counter + 1} {vertex_counter + 2}')
            vertex_counter += 2
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _rejected_bridge_sort_key(record: _RejectedBridgeRecord) -> tuple[float, int, float, tuple[int, ...]]:
    return (
        -float(record.mean_bridge_conf),
        len(record.new_edges),
        round(float(record.total_cost), 12),
        (
            int(record.source_component_id),
            -1 if record.target_component_id is None else int(record.target_component_id),
            int(record.source_vertex),
            int(record.target_vertex),
            *record.new_edges,
        ),
    )


def _write_bridge_candidates_json(
    path: Path,
    stage_e0_stats: _StageE0Stats,
    stage_b_stats: _BridgeStageStats,
    stage_c_stats: _BridgeStageStats,
    spur_stats: _SpurStageStats,
    debug_export: _BridgeDebugExport,
    rejected_limit: int = 20,
) -> None:
    rejected_rows = [
        {
            'stage': record.stage_name,
            'source_component_id': int(record.source_component_id),
            'target_component_id': None if record.target_component_id is None else int(record.target_component_id),
            'source_vertex': int(record.source_vertex),
            'target_vertex': int(record.target_vertex),
            'new_edges': [int(edge_idx) for edge_idx in record.new_edges],
            'mean_conf': float(record.mean_bridge_conf),
            'low_conf_fraction': float(record.low_conf_fraction),
            'total_cost': float(record.total_cost),
            'rejection_reason': record.rejection_reason,
        }
        for record in sorted(debug_export.rejected_bridges, key=_rejected_bridge_sort_key)[:rejected_limit]
    ]
    payload = {
        'stage_e0': _stage_e0_stats_payload(stage_e0_stats),
        'stage_b': _bridge_stage_stats_payload(stage_b_stats),
        'stage_c': _bridge_stage_stats_payload(stage_c_stats),
        'stage_spur': _spur_stage_stats_payload(spur_stats),
        'bridge_persistence_summary': {
            'accepted_bridge_edges_total': int(
                debug_export.persistence_checks.get('before_final_return', {}).get(
                    'accepted_bridge_edges_total',
                    len(debug_export.accepted_bridge_edge_order),
                )
            ),
            'accepted_bridge_edges_present_after_stage_b': int(
                debug_export.persistence_checks.get('end_of_stage_b', {}).get('accepted_bridge_edges_present', 0)
            ),
            'accepted_bridge_edges_present_after_stage_c': int(
                debug_export.persistence_checks.get('end_of_stage_c', {}).get('accepted_bridge_edges_present', 0)
            ),
            'accepted_bridge_edges_present_before_return': int(
                debug_export.persistence_checks.get('before_final_return', {}).get('accepted_bridge_edges_present', 0)
            ),
            'accepted_bridge_edges_missing_without_reason': int(
                debug_export.persistence_checks.get('before_final_return', {}).get(
                    'accepted_bridge_edges_missing_without_reason',
                    0,
                )
            ),
        },
        'terminal_groups': [
            {
                'stage': record.stage_name,
                'component_id': int(record.component_id),
                'vertex_indices': [int(vertex_idx) for vertex_idx in record.vertex_indices],
            }
            for record in debug_export.terminals
        ],
        'accepted_bridges': [
            {
                'stage': record.stage_name,
                'source_component_id': int(record.source_component_id),
                'target_component_id': None if record.target_component_id is None else int(record.target_component_id),
                'source_vertex': int(record.source_vertex),
                'target_vertex': int(record.target_vertex),
                'new_edges': [int(edge_idx) for edge_idx in record.new_edges],
            }
            for record in debug_export.accepted_bridges
        ],
        'e0_bands': [
            {
                'component_id': int(record.component_id),
                'kept_edge_ids': [int(edge_idx) for edge_idx in record.kept_edge_ids],
                'removed_edge_ids': [int(edge_idx) for edge_idx in record.removed_edge_ids],
            }
            for record in debug_export.e0_bands
        ],
        'removed_spurs': [
            {
                'source_vertex': int(record.source_vertex),
                'attach_vertex': int(record.attach_vertex),
                'chain_edges': [int(edge_idx) for edge_idx in record.chain_edges],
                'mean_conf': float(record.mean_conf),
                'added_fraction': float(record.added_fraction),
            }
            for record in debug_export.removed_spurs
        ],
        'bridge_persistence': debug_export.persistence_checks,
        'top_rejected_bridges': rejected_rows,
    }
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _write_bridge_debug_exports(
    debug_export: _BridgeDebugExport | None,
    topology: Any,
    graph: _GraphViews,
    stage_e0_stats: _StageE0Stats,
    stage_b_stats: _BridgeStageStats,
    stage_c_stats: _BridgeStageStats,
    spur_stats: _SpurStageStats,
) -> None:
    if debug_export is None:
        return
    coords = _debug_vertex_coords(graph, topology)
    _write_terminals_obj(debug_export.export_dir / 'terminals.obj', coords, debug_export.terminals)
    _write_accepted_bridges_obj(
        debug_export.export_dir / 'accepted_bridges.obj',
        coords,
        graph,
        debug_export.accepted_bridges,
    )
    _write_bridge_candidates_json(
        debug_export.export_dir / 'bridge_candidates.json',
        stage_e0_stats,
        stage_b_stats,
        stage_c_stats,
        spur_stats,
        debug_export,
    )


def _filter_bridge_candidate(
    path: _PathCandidate,
    seam_mask: np.ndarray,
    probabilities: np.ndarray,
    edge_component_ids: np.ndarray,
    stage_stats: _BridgeStageStats,
    *,
    stage_name: str,
    source_component_id: int,
    target_component_id: int | None,
    blocked_component_ids: set[int],
    max_new_edges: int,
    tau_bridge: float,
    conf_floor: float,
    max_low_conf_fraction: float,
    force_close_max_edges: int,
    debug_export: _BridgeDebugExport | None = None,
) -> _BridgeCandidate | None:
    new_edges = _path_new_edges_from_indices(path.edge_indices, seam_mask)
    if not new_edges:
        stage_stats.rejected_no_new_edges += 1
        stage_stats.rejected_force_close_empty += 1
        _record_rejected_bridge(
            debug_export,
            stage_name,
            path,
            new_edges,
            mean_bridge_conf=0.0,
            low_conf_fraction=0.0,
            rejection_reason='no_new_edges',
            source_component_id=source_component_id,
            target_component_id=target_component_id,
        )
        return None
    if len(new_edges) > max_new_edges:
        stage_stats.rejected_by_length += 1
        _record_rejected_bridge(
            debug_export,
            stage_name,
            path,
            new_edges,
            mean_bridge_conf=float(np.mean(probabilities[np.asarray(new_edges, dtype=np.int64)])),
            low_conf_fraction=float(np.mean(probabilities[np.asarray(new_edges, dtype=np.int64)] < conf_floor)),
            rejection_reason='length',
            source_component_id=source_component_id,
            target_component_id=target_component_id,
        )
        return None
    allowed_component_ids = {int(source_component_id)}
    if target_component_id is not None:
        allowed_component_ids.add(int(target_component_id))
    is_force_close = len(new_edges) <= force_close_max_edges
    if _path_uses_blocked_third_party_seam(
        path.edge_indices,
        seam_mask,
        edge_component_ids,
        allowed_component_ids,
        blocked_component_ids,
    ):
        stage_stats.rejected_by_third_party_protected += 1
        if is_force_close:
            stage_stats.rejected_force_close_third_party_protected += 1
        _record_rejected_bridge(
            debug_export,
            stage_name,
            path,
            new_edges,
            mean_bridge_conf=float(np.mean(probabilities[np.asarray(new_edges, dtype=np.int64)])),
            low_conf_fraction=float(np.mean(probabilities[np.asarray(new_edges, dtype=np.int64)] < conf_floor)),
            rejection_reason='third_party_protected',
            source_component_id=source_component_id,
            target_component_id=target_component_id,
        )
        return None
    mean_bridge_conf = float(np.mean(probabilities[np.asarray(new_edges, dtype=np.int64)]))
    low_conf_fraction = float(np.mean(probabilities[np.asarray(new_edges, dtype=np.int64)] < conf_floor))
    accepted_via_force_close = is_force_close
    if accepted_via_force_close:
        return _BridgeCandidate(
            source_component_id=int(source_component_id),
            target_component_id=None if target_component_id is None else int(target_component_id),
            source_vertex=int(path.source_vertex),
            target_vertex=int(path.target_vertex),
            edge_indices=tuple(int(edge_idx) for edge_idx in path.edge_indices),
            new_edges=tuple(int(edge_idx) for edge_idx in new_edges),
            path_key=_candidate_path_key(new_edges),
            total_cost=float(path.total_cost),
            mean_bridge_conf=mean_bridge_conf,
            low_conf_fraction=low_conf_fraction,
            accepted_via_force_close=True,
        )
    if mean_bridge_conf < tau_bridge:
        stage_stats.rejected_by_mean_conf += 1
        _record_rejected_bridge(
            debug_export,
            stage_name,
            path,
            new_edges,
            mean_bridge_conf=mean_bridge_conf,
            low_conf_fraction=low_conf_fraction,
            rejection_reason='mean_conf',
            source_component_id=source_component_id,
            target_component_id=target_component_id,
        )
        return None
    if low_conf_fraction > max_low_conf_fraction:
        stage_stats.rejected_by_low_conf_fraction += 1
        _record_rejected_bridge(
            debug_export,
            stage_name,
            path,
            new_edges,
            mean_bridge_conf=mean_bridge_conf,
            low_conf_fraction=low_conf_fraction,
            rejection_reason='low_conf_fraction',
            source_component_id=source_component_id,
            target_component_id=target_component_id,
        )
        return None
    return _BridgeCandidate(
        source_component_id=int(source_component_id),
        target_component_id=None if target_component_id is None else int(target_component_id),
        source_vertex=int(path.source_vertex),
        target_vertex=int(path.target_vertex),
        edge_indices=tuple(int(edge_idx) for edge_idx in path.edge_indices),
        new_edges=tuple(int(edge_idx) for edge_idx in new_edges),
        path_key=_candidate_path_key(new_edges),
        total_cost=float(path.total_cost),
        mean_bridge_conf=mean_bridge_conf,
        low_conf_fraction=low_conf_fraction,
        accepted_via_force_close=accepted_via_force_close,
    )


def _collapse_duplicate_bridge_paths(
    candidates: list[_BridgeCandidate],
    stage_stats: _BridgeStageStats,
    *,
    stage_name: str,
    debug_export: _BridgeDebugExport | None,
) -> list[_BridgeCandidate]:
    ranked = _rank_bridge_candidates(candidates)
    deduped: list[_BridgeCandidate] = []
    seen_path_keys: set[tuple[int, ...]] = set()
    for candidate in ranked:
        if candidate.path_key in seen_path_keys:
            stage_stats.duplicate_paths_collapsed += 1
            _record_rejected_bridge(
                debug_export,
                stage_name,
                _PathCandidate(
                    source_vertex=int(candidate.source_vertex),
                    target_vertex=int(candidate.target_vertex),
                    edge_indices=tuple(int(edge_idx) for edge_idx in candidate.edge_indices),
                    total_cost=float(candidate.total_cost),
                    total_edges=len(candidate.edge_indices),
                    normalized_cost=float(candidate.total_cost / max(len(candidate.edge_indices), 1)),
                ),
                candidate.new_edges,
                mean_bridge_conf=float(candidate.mean_bridge_conf),
                low_conf_fraction=float(candidate.low_conf_fraction),
                rejection_reason='duplicate_of_existing_path',
                source_component_id=int(candidate.source_component_id),
                target_component_id=int(candidate.target_component_id),
            )
            continue
        seen_path_keys.add(candidate.path_key)
        deduped.append(candidate)
    return deduped


def _find_ambiguity_competitor(
    best_candidate: _BridgeCandidate,
    candidates: list[_BridgeCandidate],
    *,
    ambiguity_same_path_jaccard: float,
) -> _BridgeCandidate | None:
    for candidate in candidates[1:]:
        if candidate.source_component_id != best_candidate.source_component_id:
            continue
        if candidate.path_key == best_candidate.path_key:
            continue
        if _candidate_edge_jaccard(best_candidate, candidate) >= ambiguity_same_path_jaccard:
            continue
        return candidate
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
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    if component.edge_count > snap_max_edges:
        return mask.copy(), (), ()
    if frozenset(component.edge_indices) in preserved_loops:
        return mask.copy(), (), ()

    distance_map = _edge_distance_map(main_component.edge_indices, graph.edge_neighbors)
    distance_to_main = _component_distance_to_edges(component, distance_map)
    if distance_to_main is None or distance_to_main > r_snap:
        return mask.copy(), (), ()

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
        return mask.copy(), (), ()

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
        return mask.copy(), (), ()

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
    return out, tuple(sorted(removed)), tuple(sorted(int(edge_idx) for edge_idx in backbone_edges))


def _safe_loop_signature(component: _SeamComponent) -> frozenset[int]:
    return frozenset(int(edge_idx) for edge_idx in component.edge_indices)


def _is_protected_component(
    component: _SeamComponent,
    main_component: _SeamComponent | None,
    *,
    protect_min_edges: int,
    protect_min_mass: float,
    protect_rel_frac: float,
) -> bool:
    if component.edge_count >= protect_min_edges:
        return True
    if component.seam_mass >= protect_min_mass:
        return True
    if (
        main_component is not None
        and main_component.edge_count >= protect_min_edges
        and component.edge_count >= protect_rel_frac * main_component.edge_count
    ):
        return True
    return False


def _is_garbage_component(
    component: _SeamComponent,
    main_component: _SeamComponent | None,
    failed_self_bridges: set[frozenset[int]],
    failed_cross_bridges: set[frozenset[int]],
    *,
    protect_min_edges: int,
    protect_min_mass: float,
    protect_rel_frac: float,
    garbage_max_edges: int,
    garbage_max_mass: float,
) -> bool:
    signature = _safe_loop_signature(component)
    if not component.is_open:
        return False
    if main_component is not None and signature == _safe_loop_signature(main_component):
        return False
    if _is_protected_component(
        component,
        main_component,
        protect_min_edges=protect_min_edges,
        protect_min_mass=protect_min_mass,
        protect_rel_frac=protect_rel_frac,
    ):
        return False
    if component.edge_count > garbage_max_edges:
        return False
    if component.seam_mass > garbage_max_mass:
        return False
    if component.cycle_rank != 0:
        return False
    return signature in failed_self_bridges and signature in failed_cross_bridges


def _log_stage_summary(
    stage_name: str,
    mask: np.ndarray,
    graph: _GraphViews,
    probabilities: np.ndarray,
    *,
    deleted_components: int,
    deleted_edges: int,
    protect_min_edges: int,
    protect_min_mass: float,
    protect_rel_frac: float,
) -> None:
    if not _LOGGER.isEnabledFor(logging.DEBUG):
        return
    components = _analyze_components(mask, graph, probabilities)
    main_component = _choose_main_open_component(components)
    protected_open_count = sum(
        1
        for component in components
        if component.is_open
        and _is_protected_component(
            component,
            main_component,
            protect_min_edges=protect_min_edges,
            protect_min_mass=protect_min_mass,
            protect_rel_frac=protect_rel_frac,
        )
    )
    _LOGGER.debug(
        'Postprocess %s: seam_edges=%d components=%d protected_open=%d deleted_components=%d deleted_edges=%d',
        stage_name,
        int(mask.sum()),
        len(components),
        protected_open_count,
        int(deleted_components),
        int(deleted_edges),
    )


def apply_seam_postprocessing_detailed(
    topology: Any,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
    max_gap_length: int = 8,
    min_island_size: int = 3,
    *,
    seam_threshold: float | None = None,
    alpha_cost: float = 0.5,
    tau_bridge: float = 0.20,
    conf_floor: float = 0.10,
    max_low_conf_fraction: float = 0.50,
    force_close_max_edges: int = 5,
    lambda_off: float = 0.75,
    r_self: int = 8,
    r_cross: int = 10,
    ambiguity_margin: float = 0.05,
    ambiguity_same_path_jaccard: float = 0.8,
    tau_path: float = 1.35,
    kappa_self: float = 1.5,
    attach_margin: float = 0.10,
    protect_min_edges: int = 12,
    protect_min_mass: float = 6.0,
    protect_rel_frac: float = 0.20,
    garbage_max_edges: int = 5,
    garbage_max_mass: float = 2.5,
    e0_radius: int = 2,
    e0_length_penalty: float = 0.05,
    r_snap: int = 3,
    snap_max_edges: int = 10,
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
    max_spur_edges: int = 3,
    spur_mean_conf: float | None = None,
    spur_mean_max: float = 0.50,
    spur_added_fraction_min: float = 0.50,
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
    debug_export_dir: str | Path | None = None,
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
    spur_mean_conf_value = float(spur_mean_max if spur_mean_conf is None else spur_mean_conf)
    _validate_probability_threshold('threshold', seam_threshold_value)
    _validate_probability_threshold('tau_bridge', float(tau_bridge))
    _validate_probability_threshold('conf_floor', float(conf_floor))
    _validate_probability_threshold('spur_mean_conf', spur_mean_conf_value)
    if alpha_cost < 0.0:
        raise ValueError(f'alpha_cost must be non-negative, got {alpha_cost}')
    if force_close_max_edges < 0:
        raise ValueError(f'force_close_max_edges must be non-negative, got {force_close_max_edges}')
    if lambda_off < 0.0:
        raise ValueError(f'lambda_off must be non-negative, got {lambda_off}')
    if r_self < 0 or r_cross < 0:
        raise ValueError(f'r_self and r_cross must be non-negative, got {r_self}, {r_cross}')
    if ambiguity_margin < 0.0:
        raise ValueError(f'ambiguity_margin must be non-negative, got {ambiguity_margin}')
    if ambiguity_same_path_jaccard < 0.0 or ambiguity_same_path_jaccard > 1.0:
        raise ValueError(
            f'ambiguity_same_path_jaccard must be a finite value in [0, 1], got {ambiguity_same_path_jaccard}'
        )
    if tau_path < 0.0:
        raise ValueError(f'tau_path must be non-negative, got {tau_path}')
    if kappa_self < 0.0:
        raise ValueError(f'kappa_self must be non-negative, got {kappa_self}')
    if attach_margin < 0.0:
        raise ValueError(f'attach_margin must be non-negative, got {attach_margin}')
    if max_low_conf_fraction < 0.0 or max_low_conf_fraction > 1.0:
        raise ValueError(
            f'max_low_conf_fraction must be a finite value in [0, 1], got {max_low_conf_fraction}'
        )
    if protect_min_edges < 0:
        raise ValueError(f'protect_min_edges must be non-negative, got {protect_min_edges}')
    if protect_min_mass < 0.0:
        raise ValueError(f'protect_min_mass must be non-negative, got {protect_min_mass}')
    if protect_rel_frac < 0.0:
        raise ValueError(f'protect_rel_frac must be non-negative, got {protect_rel_frac}')
    if garbage_max_mass < 0.0:
        raise ValueError(f'garbage_max_mass must be non-negative, got {garbage_max_mass}')
    if garbage_max_edges < 0 or e0_radius < 0 or r_snap < 0 or snap_max_edges < 0 or r_band < 0:
        raise ValueError('graph-radius parameters must be non-negative')
    if e0_length_penalty < 0.0:
        raise ValueError(f'e0_length_penalty must be non-negative, got {e0_length_penalty}')
    if eta_main < 0.0:
        raise ValueError(f'eta_main must be non-negative, got {eta_main}')
    if max_spur_edges < 0:
        raise ValueError(f'max_spur_edges must be non-negative, got {max_spur_edges}')
    if spur_added_fraction_min < 0.0 or spur_added_fraction_min > 1.0:
        raise ValueError(
            f'spur_added_fraction_min must be a finite value in [0, 1], got {spur_added_fraction_min}'
        )

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
    bridge_edge_costs = _compute_search_edge_costs(probs, float(alpha_cost))
    edge_costs = _edge_costs(probs, seam_threshold_value, float(lambda_off))
    debug_export = _make_debug_export(debug_export_dir)

    threshold_mask = probs >= seam_threshold_value
    working_seam_mask = threshold_mask.copy()
    added_bridge_edges: set[int] = set()
    stage_b_accepted_bridge_edges: set[int] = set()
    preserved_loops: set[frozenset[int]] = set()
    failed_self_bridges: set[frozenset[int]] = set()
    failed_cross_bridges: set[frozenset[int]] = set()
    pruned_edges: set[int] = set()
    pruned_component_count = 0
    edge_origin = np.full(len(edges), 'threshold', dtype=object)

    _log_stage_summary(
        'threshold',
        working_seam_mask,
        graph,
        probs,
        deleted_components=0,
        deleted_edges=0,
        protect_min_edges=int(protect_min_edges),
        protect_min_mass=float(protect_min_mass),
        protect_rel_frac=float(protect_rel_frac),
    )

    # Stage E0
    working_seam_mask, stage_e0_stats = _apply_stage_e0(
        working_seam_mask,
        graph,
        probs,
        edge_origin,
        e0_radius=int(e0_radius),
        e0_length_penalty=float(e0_length_penalty),
        debug_export=debug_export,
    )
    _log_stage_e0_stats(stage_e0_stats)
    _log_stage_summary(
        'pre_bridge_thickness_collapse',
        working_seam_mask,
        graph,
        probs,
        deleted_components=0,
        deleted_edges=int(stage_e0_stats.e0_edges_removed),
        protect_min_edges=int(protect_min_edges),
        protect_min_mass=float(protect_min_mass),
        protect_rel_frac=float(protect_rel_frac),
    )

    # Stage B
    stage_b_stats = _BridgeStageStats()
    initial_components = _analyze_components(working_seam_mask, graph, probs)
    initial_edge_component_ids = _build_edge_component_ids(initial_components, graph.edge_count)
    main_component = _choose_main_open_component(initial_components)
    main_component_id = None if main_component is None else int(main_component.component_id)
    stage_b_blocked_component_ids = {
        int(component.component_id)
        for component in initial_components
        if component.component_id == main_component_id
        or _is_protected_component(
            component,
            main_component,
            protect_min_edges=int(protect_min_edges),
            protect_min_mass=float(protect_min_mass),
            protect_rel_frac=float(protect_rel_frac),
        )
    }
    eligible_components = [
        component
        for component in initial_components
        if component.is_open and len(component.vertex_indices) >= 2
    ]
    eligible_components.sort(
        key=lambda component: (
            int(component.component_id == main_component_id),
            int(component.edge_count),
            int(component.component_id),
        )
    )
    for component in eligible_components:
        signature = _safe_loop_signature(component)
        seam_vertex_degrees = _component_vertex_degrees(component.edge_indices, graph)
        terminals = _build_terminal_set(component, seam_vertex_degrees)
        _record_terminals(debug_export, 'stage_b', int(component.component_id), terminals)
        if len(terminals) < 2:
            failed_self_bridges.add(signature)
            continue
        valid_candidates: list[_BridgeCandidate] = []
        for index, source_vertex in enumerate(terminals):
            for target_vertex in terminals[index + 1:]:
                stage_b_stats.candidate_pairs_considered += 1
                paths = _bounded_shortest_path(
                    graph,
                    bridge_edge_costs,
                    int(source_vertex),
                    {int(target_vertex)},
                    max_edges=int(r_self),
                )
                if not paths:
                    continue
                stage_b_stats.shortest_paths_found += 1
                candidate = _filter_bridge_candidate(
                    paths[0],
                    working_seam_mask,
                    probs,
                    initial_edge_component_ids,
                    stage_b_stats,
                    stage_name='stage_b',
                    source_component_id=int(component.component_id),
                    target_component_id=int(component.component_id),
                    blocked_component_ids=stage_b_blocked_component_ids,
                    max_new_edges=int(r_self),
                    tau_bridge=float(tau_bridge),
                    conf_floor=float(conf_floor),
                    max_low_conf_fraction=float(max_low_conf_fraction),
                    force_close_max_edges=int(force_close_max_edges),
                    debug_export=debug_export,
                )
                if candidate is not None:
                    valid_candidates.append(candidate)
        if not valid_candidates:
            failed_self_bridges.add(signature)
            continue
        ranked_candidates = _collapse_duplicate_bridge_paths(
            valid_candidates,
            stage_b_stats,
            stage_name='stage_b',
            debug_export=debug_export,
        )
        if not ranked_candidates:
            failed_self_bridges.add(signature)
            continue
        force_close_candidates = [candidate for candidate in ranked_candidates if candidate.accepted_via_force_close]
        best_candidate = force_close_candidates[0] if force_close_candidates else ranked_candidates[0]
        ambiguity_competitor = _find_ambiguity_competitor(
            best_candidate,
            ranked_candidates,
            ambiguity_same_path_jaccard=float(ambiguity_same_path_jaccard),
        )
        if (
            not best_candidate.accepted_via_force_close
            and
            ambiguity_competitor is not None
            and best_candidate.mean_bridge_conf < ambiguity_competitor.mean_bridge_conf + float(ambiguity_margin)
        ):
            stage_b_stats.rejected_by_ambiguity += 1
            _record_rejected_bridge(
                debug_export,
                'stage_b',
                _PathCandidate(
                    source_vertex=int(best_candidate.source_vertex),
                    target_vertex=int(best_candidate.target_vertex),
                    edge_indices=tuple(int(edge_idx) for edge_idx in best_candidate.edge_indices),
                    total_cost=float(best_candidate.total_cost),
                    total_edges=len(best_candidate.edge_indices),
                    normalized_cost=float(best_candidate.total_cost / max(len(best_candidate.edge_indices), 1)),
                ),
                best_candidate.new_edges,
                mean_bridge_conf=float(best_candidate.mean_bridge_conf),
                low_conf_fraction=float(best_candidate.low_conf_fraction),
                rejection_reason='ambiguity',
                source_component_id=int(best_candidate.source_component_id),
                target_component_id=int(best_candidate.target_component_id),
            )
            failed_self_bridges.add(signature)
            continue
        for edge_idx in best_candidate.new_edges:
            working_seam_mask[int(edge_idx)] = True
            edge_origin[int(edge_idx)] = 'bridge_b'
            added_bridge_edges.add(int(edge_idx))
            stage_b_accepted_bridge_edges.add(int(edge_idx))
        _record_accepted_bridge_edges(debug_export, best_candidate.new_edges)
        stage_b_stats.accepted_bridges += 1
        if best_candidate.accepted_via_force_close:
            stage_b_stats.accepted_via_force_close += 1
        stage_b_stats.total_new_edges_added += len(best_candidate.new_edges)
        _record_accepted_bridge(debug_export, 'stage_b', best_candidate)
        updated_components = _analyze_components(working_seam_mask, graph, probs)
        for updated in updated_components:
            if set(component.edge_indices).issubset(set(updated.edge_indices)) and updated.is_closed:
                preserved_loops.add(_safe_loop_signature(updated))
                break

    stage_b_stats.total_components_merged += max(
        len(initial_components) - len(_analyze_components(working_seam_mask, graph, probs)),
        0,
    )
    _record_bridge_persistence_checkpoint(debug_export, 'end_of_stage_b', working_seam_mask, stage_b_accepted_bridge_edges)
    stage_b_mask = working_seam_mask.copy()
    _log_bridge_stats('self_bridge', stage_b_stats)
    _log_stage_summary(
        'self_bridge',
        working_seam_mask,
        graph,
        probs,
        deleted_components=0,
        deleted_edges=0,
        protect_min_edges=int(protect_min_edges),
        protect_min_mass=float(protect_min_mass),
        protect_rel_frac=float(protect_rel_frac),
    )

    # Stage C
    stage_c_stats = _BridgeStageStats()
    bridge_count = int(stage_b_stats.accepted_bridges)
    iteration_cap = 8
    for _iteration in range(iteration_cap):
        components = _analyze_components(working_seam_mask, graph, probs)
        main_component = _choose_main_open_component(components)
        if main_component is None:
            break
        protected_target_ids = {
            int(component.component_id)
            for component in components
            if _is_protected_component(
                component,
                main_component,
                protect_min_edges=int(protect_min_edges),
                protect_min_mass=float(protect_min_mass),
                protect_rel_frac=float(protect_rel_frac),
            )
        }
        protected_target_ids.add(int(main_component.component_id))
        if len(protected_target_ids) == 0:
            break
        blocked_component_ids = set(int(component_id) for component_id in protected_target_ids)
        edge_component_ids = _build_edge_component_ids(components, graph.edge_count)
        target_terminals: dict[int, tuple[int, ...]] = {}
        for component in components:
            if component.component_id not in protected_target_ids:
                continue
            component_terminals = _build_terminal_set(
                component,
                _component_vertex_degrees(component.edge_indices, graph),
            )
            target_terminals[int(component.component_id)] = component_terminals
            _record_terminals(debug_export, 'stage_c', int(component.component_id), component_terminals)

        component_candidates: list[_BridgeCandidate] = []
        component_count_before = len(components)
        for component in components:
            if component.component_id == main_component.component_id or not component.is_open:
                continue
            signature = _safe_loop_signature(component)
            source_terminals = _build_terminal_set(component, _component_vertex_degrees(component.edge_indices, graph))
            _record_terminals(debug_export, 'stage_c', int(component.component_id), source_terminals)
            if len(source_terminals) == 0:
                failed_cross_bridges.add(signature)
                continue
            valid_candidates: list[_BridgeCandidate] = []
            for target_component in components:
                target_component_id = int(target_component.component_id)
                if target_component_id == int(component.component_id) or target_component_id not in protected_target_ids:
                    continue
                for source_vertex in source_terminals:
                    for target_vertex in target_terminals.get(target_component_id, ()):
                        stage_c_stats.candidate_pairs_considered += 1
                        paths = _bounded_shortest_path(
                            graph,
                            bridge_edge_costs,
                            int(source_vertex),
                            {int(target_vertex)},
                            max_edges=int(r_cross),
                        )
                        if not paths:
                            continue
                        stage_c_stats.shortest_paths_found += 1
                        candidate = _filter_bridge_candidate(
                            paths[0],
                            working_seam_mask,
                            probs,
                            edge_component_ids,
                            stage_c_stats,
                            stage_name='stage_c',
                            source_component_id=int(component.component_id),
                            target_component_id=int(target_component_id),
                            blocked_component_ids=blocked_component_ids,
                            max_new_edges=int(r_cross),
                            tau_bridge=float(tau_bridge),
                            conf_floor=float(conf_floor),
                            max_low_conf_fraction=float(max_low_conf_fraction),
                            force_close_max_edges=int(force_close_max_edges),
                            debug_export=debug_export,
                        )
                        if candidate is not None:
                            valid_candidates.append(candidate)
            if not valid_candidates:
                failed_cross_bridges.add(signature)
                continue
            ranked_candidates = _collapse_duplicate_bridge_paths(
                valid_candidates,
                stage_c_stats,
                stage_name='stage_c',
                debug_export=debug_export,
            )
            if not ranked_candidates:
                failed_cross_bridges.add(signature)
                continue
            force_close_candidates = [candidate for candidate in ranked_candidates if candidate.accepted_via_force_close]
            best_candidate = force_close_candidates[0] if force_close_candidates else ranked_candidates[0]
            ambiguity_competitor = _find_ambiguity_competitor(
                best_candidate,
                ranked_candidates,
                ambiguity_same_path_jaccard=float(ambiguity_same_path_jaccard),
            )
            if (
                not best_candidate.accepted_via_force_close
                and
                ambiguity_competitor is not None
                and best_candidate.mean_bridge_conf < ambiguity_competitor.mean_bridge_conf + float(ambiguity_margin)
            ):
                stage_c_stats.rejected_by_ambiguity += 1
                _record_rejected_bridge(
                    debug_export,
                    'stage_c',
                    _PathCandidate(
                        source_vertex=int(best_candidate.source_vertex),
                        target_vertex=int(best_candidate.target_vertex),
                        edge_indices=tuple(int(edge_idx) for edge_idx in best_candidate.edge_indices),
                        total_cost=float(best_candidate.total_cost),
                        total_edges=len(best_candidate.edge_indices),
                        normalized_cost=float(best_candidate.total_cost / max(len(best_candidate.edge_indices), 1)),
                    ),
                    best_candidate.new_edges,
                    mean_bridge_conf=float(best_candidate.mean_bridge_conf),
                    low_conf_fraction=float(best_candidate.low_conf_fraction),
                    rejection_reason='ambiguity',
                    source_component_id=int(best_candidate.source_component_id),
                    target_component_id=int(best_candidate.target_component_id),
                )
                failed_cross_bridges.add(signature)
                continue
            component_candidates.append(best_candidate)
        if not component_candidates:
            break
        component_candidates = _rank_bridge_candidates(component_candidates)
        accepted_sources: set[int] = set()
        any_accepted = False
        for candidate in component_candidates:
            if candidate.source_component_id in accepted_sources:
                continue
            accepted_sources.add(int(candidate.source_component_id))
            for edge_idx in candidate.new_edges:
                working_seam_mask[int(edge_idx)] = True
                edge_origin[int(edge_idx)] = 'bridge_c'
                added_bridge_edges.add(int(edge_idx))
            _record_accepted_bridge_edges(debug_export, candidate.new_edges)
            stage_c_stats.accepted_bridges += 1
            if candidate.accepted_via_force_close:
                stage_c_stats.accepted_via_force_close += 1
            stage_c_stats.total_new_edges_added += len(candidate.new_edges)
            _record_accepted_bridge(debug_export, 'stage_c', candidate)
            bridge_count += 1
            any_accepted = True
            break
        if not any_accepted:
            break
        component_count_after = len(_analyze_components(working_seam_mask, graph, probs))
        stage_c_stats.total_components_merged += max(component_count_before - component_count_after, 0)

    _record_bridge_persistence_checkpoint(debug_export, 'end_of_stage_c', working_seam_mask, added_bridge_edges)
    _log_bridge_stats('cross_bridge', stage_c_stats)
    _log_stage_summary(
        'cross_bridge',
        working_seam_mask,
        graph,
        probs,
        deleted_components=0,
        deleted_edges=0,
        protect_min_edges=int(protect_min_edges),
        protect_min_mass=float(protect_min_mass),
        protect_rel_frac=float(protect_rel_frac),
    )

    # Stage D
    components_after_cross = _analyze_components(working_seam_mask, graph, probs)
    main_component = _choose_main_open_component(components_after_cross)
    stage_d_deleted_components = 0
    stage_d_deleted_edges = 0
    for component in components_after_cross:
        if not _is_garbage_component(
            component,
            main_component,
            failed_self_bridges,
            failed_cross_bridges,
            protect_min_edges=int(protect_min_edges),
            protect_min_mass=float(protect_min_mass),
            protect_rel_frac=float(protect_rel_frac),
            garbage_max_edges=int(garbage_max_edges),
            garbage_max_mass=float(garbage_max_mass),
        ):
            continue
        working_seam_mask[np.asarray(component.edge_indices, dtype=np.int64)] = False
        _mark_removed_bridge_edges(debug_export, component.edge_indices, 'removed_by_other_named_stage')
        pruned_edges.update(int(edge_idx) for edge_idx in component.edge_indices)
        pruned_component_count += 1
        stage_d_deleted_components += 1
        stage_d_deleted_edges += component.edge_count

    _log_stage_summary(
        'garbage_collect',
        working_seam_mask,
        graph,
        probs,
        deleted_components=stage_d_deleted_components,
        deleted_edges=stage_d_deleted_edges,
        protect_min_edges=int(protect_min_edges),
        protect_min_mass=float(protect_min_mass),
        protect_rel_frac=float(protect_rel_frac),
    )

    # Stage Spur
    working_seam_mask, spur_stats, spur_removed_edges = _apply_spur_cleanup(
        working_seam_mask,
        graph,
        probs,
        edge_origin,
        max_spur_edges=int(max_spur_edges),
        spur_mean_conf=spur_mean_conf_value,
        spur_added_fraction_min=float(spur_added_fraction_min),
        debug_export=debug_export,
    )
    if spur_removed_edges:
        _mark_removed_bridge_edges(debug_export, spur_removed_edges, 'removed_by_spur_cleanup')
        pruned_edges.update(int(edge_idx) for edge_idx in spur_removed_edges)
    _log_spur_stats(spur_stats)
    _log_stage_summary(
        'spur_cleanup',
        working_seam_mask,
        graph,
        probs,
        deleted_components=0,
        deleted_edges=int(spur_stats.spur_edges_removed),
        protect_min_edges=int(protect_min_edges),
        protect_min_mass=float(protect_min_mass),
        protect_rel_frac=float(protect_rel_frac),
    )

    # Stage E
    components_before_snap = _analyze_components(working_seam_mask, graph, probs)
    main_component = _choose_main_open_component(components_before_snap)
    stage_e_deleted_edges = 0
    if main_component is not None:
        for component in components_before_snap:
            if component.component_id == main_component.component_id:
                continue
            if _is_protected_component(
                component,
                main_component,
                protect_min_edges=int(protect_min_edges),
                protect_min_mass=float(protect_min_mass),
                protect_rel_frac=float(protect_rel_frac),
            ):
                continue
            working_seam_mask, removed, kept_backbone = _apply_band_collapse(
                working_seam_mask,
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
                _mark_removed_bridge_edges(debug_export, removed, 'removed_by_stage_e')
                pruned_edges.update(int(edge_idx) for edge_idx in removed)
                stage_e_deleted_edges += len(removed)
            for edge_idx in kept_backbone:
                edge_origin[int(edge_idx)] = 'stage_e'

    _log_stage_summary(
        'band_collapse',
        working_seam_mask,
        graph,
        probs,
        deleted_components=0,
        deleted_edges=stage_e_deleted_edges,
        protect_min_edges=int(protect_min_edges),
        protect_min_mass=float(protect_min_mass),
        protect_rel_frac=float(protect_rel_frac),
    )

    # Stage F
    final_components = _analyze_components(working_seam_mask, graph, probs)
    main_component = _choose_main_open_component(final_components)
    stage_f_deleted_components = 0
    stage_f_deleted_edges = 0
    for component in final_components:
        if not _is_garbage_component(
            component,
            main_component,
            failed_self_bridges,
            failed_cross_bridges,
            protect_min_edges=int(protect_min_edges),
            protect_min_mass=float(protect_min_mass),
            protect_rel_frac=float(protect_rel_frac),
            garbage_max_edges=int(garbage_max_edges),
            garbage_max_mass=float(garbage_max_mass),
        ):
            continue
        working_seam_mask[np.asarray(component.edge_indices, dtype=np.int64)] = False
        _mark_removed_bridge_edges(debug_export, component.edge_indices, 'removed_by_final_cleanup')
        pruned_edges.update(int(edge_idx) for edge_idx in component.edge_indices)
        pruned_component_count += 1
        stage_f_deleted_components += 1
        stage_f_deleted_edges += component.edge_count

    final_components = _analyze_components(working_seam_mask, graph, probs)
    open_main = _choose_main_open_component(final_components)
    _record_bridge_persistence_checkpoint(debug_export, 'before_final_return', working_seam_mask, added_bridge_edges)
    _write_bridge_debug_exports(
        debug_export,
        topology,
        graph,
        stage_e0_stats,
        stage_b_stats,
        stage_c_stats,
        spur_stats,
    )
    _log_stage_summary(
        'final_cleanup',
        working_seam_mask,
        graph,
        probs,
        deleted_components=stage_f_deleted_components,
        deleted_edges=stage_f_deleted_edges,
        protect_min_edges=int(protect_min_edges),
        protect_min_mass=float(protect_min_mass),
        protect_rel_frac=float(protect_rel_frac),
    )
    endpoint_count = 0 if open_main is None else len(open_main.endpoint_vertices)

    bridge_mask = np.zeros(len(edges), dtype=bool)
    if added_bridge_edges:
        bridge_mask[np.asarray(sorted(added_bridge_edges), dtype=np.int64)] = True

    return SeamPostprocessResult(
        threshold_mask=threshold_mask,
        skeleton_mask=stage_b_mask,
        steiner_mask=bridge_mask,
        final_mask=working_seam_mask.copy(),
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
