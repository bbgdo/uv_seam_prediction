from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Any

import networkx as nx
import numpy as np

from models.utils.seam_topology_diagnostics import SeamMaskDiagnostics, compute_seam_mask_diagnostics
from models.utils.seam_topology_view import SeamGraphView
from preprocessing.topology import CanonicalTopology

BRIDGE_MAX_ENDPOINT_CANDIDATES = 4
BRIDGE_REQUIRE_MUTUAL_PAIRING = True
BRIDGE_TANGENT_ALIGNMENT_WEIGHT = 0.25
BRIDGE_MAX_DEBUG_CANDIDATES = 64


@dataclass(frozen=True)
class SkeletonResult:
    initial_candidate_vertices: frozenset[int]
    anchor_vertices: frozenset[int]
    skeleton_vertices: frozenset[int]
    skeleton_edge_mask: np.ndarray
    vertex_scores: np.ndarray
    iterations_performed: int
    removals_committed: int
    refused_by_anchor: int
    refused_by_simple_test: int
    refused_by_distance_test: int
    tau_low: float
    d_max: int
    anchor_boundary: bool


def boundary_vertices_from_topology(topology: CanonicalTopology | None) -> frozenset[int]:
    if topology is None:
        return frozenset()

    boundary_vertices: set[int] = set()
    for edge_key, occurrences in topology.edge_incidence.items():
        if len(occurrences) == 1:
            boundary_vertices.add(int(edge_key[0]))
            boundary_vertices.add(int(edge_key[1]))
    return frozenset(boundary_vertices)


def lift_edge_probabilities_to_vertices(
    view: SeamGraphView,
    probabilities: np.ndarray,
) -> np.ndarray:
    probs = _validated_probability_vector(view, probabilities)
    vertex_scores = np.zeros(view.vertex_count, dtype=np.float64)
    for vertex_index, edge_indices in enumerate(view.vertex_to_edges):
        if not edge_indices:
            continue
        vertex_scores[vertex_index] = float(np.max(probs[np.asarray(edge_indices, dtype=np.int64)]))
    return vertex_scores


def compute_topology_preserving_skeleton(
    view: SeamGraphView,
    probabilities: np.ndarray,
    *,
    tau_low: float = 0.30,
    d_max: int = 3,
    anchor_boundary: bool = True,
    extra_anchor_vertices: frozenset[int] | None = None,
    topology: Any = None,
) -> SkeletonResult:
    probs = _validated_probability_vector(view, probabilities)
    tau_low_value = _validated_probability_threshold('tau_low', tau_low)
    if isinstance(d_max, bool) or not isinstance(d_max, (int, np.integer)) or int(d_max) < 1:
        raise ValueError('d_max must be an integer greater than or equal to 1')
    d_max_value = int(d_max)
    if anchor_boundary and topology is None:
        raise ValueError('anchor_boundary=True requires a non-None topology argument')

    normalized_extra_anchors: frozenset[int] | None = None
    if extra_anchor_vertices is not None:
        normalized_anchor_vertices: set[int] = set()
        for vertex in extra_anchor_vertices:
            if isinstance(vertex, bool) or not isinstance(vertex, (int, np.integer)):
                raise ValueError('extra_anchor_vertices must contain integer vertex indices')
            vertex_index = int(vertex)
            if vertex_index < 0 or vertex_index >= view.vertex_count:
                raise ValueError(
                    f'extra_anchor_vertices contains out-of-range vertex index {vertex_index} '
                    f'for vertex_count={view.vertex_count}'
                )
            normalized_anchor_vertices.add(vertex_index)
        normalized_extra_anchors = frozenset(normalized_anchor_vertices)

    vertex_scores = lift_edge_probabilities_to_vertices(view, probs)

    C = {
        int(vertex_index)
        for vertex_index in np.flatnonzero(vertex_scores >= tau_low_value)
    }
    initial_C = frozenset(C)
    in_C = np.zeros(view.vertex_count, dtype=bool)
    if C:
        in_C[np.asarray(sorted(C), dtype=np.int64)] = True

    A: set[int] = set()
    if anchor_boundary:
        A.update(boundary_vertices_from_topology(topology))
    if normalized_extra_anchors is not None:
        A.update(int(vertex) for vertex in normalized_extra_anchors)
    A = {vertex for vertex in A if vertex in C}
    in_A = np.zeros(view.vertex_count, dtype=bool)
    for vertex in A:
        in_A[vertex] = True

    heap: list[tuple[float, int]] = [
        (float(vertex_scores[vertex_index]), int(vertex_index))
        for vertex_index in sorted(C)
        if not in_A[vertex_index]
    ]
    heapq.heapify(heap)

    D: set[int] = set()
    nearest_dist: dict[int, int] = {}
    adjacency = view.vertex_graph.adj

    iterations_performed = 0
    removals_committed = 0
    refused_by_anchor = 0
    refused_by_simple_test = 0
    refused_by_distance_test = 0

    while heap:
        score_v, vertex = heapq.heappop(heap)
        iterations_performed += 1

        if not in_C[vertex]:
            continue
        if in_A[vertex]:
            refused_by_anchor += 1
            continue

        del score_v
        if not _passes_simple_vertex_test(adjacency, vertex, in_C, depth_bound=(2 * d_max_value) + 2):
            refused_by_simple_test += 1
            continue

        distance_to_candidates = _bounded_distance_to_candidate_set(
            adjacency,
            vertex,
            in_C,
            max_distance=d_max_value,
            excluded_candidate=vertex,
        )
        if distance_to_candidates is None:
            refused_by_distance_test += 1
            continue

        affected_deleted_vertices = _deleted_vertices_within_radius(
            adjacency,
            vertex,
            D,
            radius=d_max_value + 1,
        )
        updated_nearest_dist: dict[int, int] = {}
        distance_test_failed = False
        for deleted_vertex in sorted(affected_deleted_vertices):
            deleted_distance = _bounded_distance_to_candidate_set(
                adjacency,
                deleted_vertex,
                in_C,
                max_distance=d_max_value,
                excluded_candidate=vertex,
            )
            if deleted_distance is None:
                distance_test_failed = True
                break
            updated_nearest_dist[deleted_vertex] = deleted_distance
        if distance_test_failed:
            refused_by_distance_test += 1
            continue

        C.remove(vertex)
        in_C[vertex] = False
        D.add(vertex)
        nearest_dist[vertex] = distance_to_candidates
        nearest_dist.update(updated_nearest_dist)
        removals_committed += 1

    skeleton_edge_mask = np.zeros(view.edge_count, dtype=bool)
    for edge_index in range(view.edge_count):
        vi = int(view.unique_edges[edge_index, 0])
        vj = int(view.unique_edges[edge_index, 1])
        if in_C[vi] and in_C[vj] and probs[edge_index] >= tau_low_value:
            skeleton_edge_mask[edge_index] = True

    return SkeletonResult(
        initial_candidate_vertices=initial_C,
        anchor_vertices=frozenset(A),
        skeleton_vertices=frozenset(C),
        skeleton_edge_mask=skeleton_edge_mask,
        vertex_scores=vertex_scores,
        iterations_performed=iterations_performed,
        removals_committed=removals_committed,
        refused_by_anchor=refused_by_anchor,
        refused_by_simple_test=refused_by_simple_test,
        refused_by_distance_test=refused_by_distance_test,
        tau_low=tau_low_value,
        d_max=d_max_value,
        anchor_boundary=bool(anchor_boundary),
    )


def diagnose_skeleton_application(
    view: SeamGraphView,
    probabilities: np.ndarray,
    *,
    tau_low: float = 0.30,
    d_max: int = 3,
    anchor_boundary: bool = True,
    extra_anchor_vertices: frozenset[int] | None = None,
    topology: Any = None,
    diagnostics_threshold: float | None = None,
) -> tuple[SkeletonResult, SeamMaskDiagnostics, SeamMaskDiagnostics]:
    threshold_value = tau_low if diagnostics_threshold is None else diagnostics_threshold
    before = compute_seam_mask_diagnostics(view, probabilities, threshold=threshold_value)
    skeleton = compute_topology_preserving_skeleton(
        view,
        probabilities,
        tau_low=tau_low,
        d_max=d_max,
        anchor_boundary=anchor_boundary,
        extra_anchor_vertices=extra_anchor_vertices,
        topology=topology,
    )
    probs_after = np.where(skeleton.skeleton_edge_mask, 1.0, 0.0).astype(np.float64, copy=False)
    after = compute_seam_mask_diagnostics(view, probs_after, threshold=0.5)
    return skeleton, before, after


@dataclass(frozen=True)
class BridgingResult:
    bridged_edge_mask: np.ndarray
    added_bridge_edges: frozenset[int] = frozenset()
    endpoints_before: int = 0
    endpoints_after: int = 0
    components_before: int = 0
    components_after: int = 0
    endpoint_pairs_considered: int = 0
    candidate_paths_found: int = 0
    candidate_paths_valid: int = 0
    bridges_accepted: int = 0
    added_bridge_edges_count: int = 0
    max_bridge_length_edges: int = 0
    mean_bridge_length_edges: float = 0.0
    bridges_rejected_by_no_path: int = 0
    bridges_rejected_by_graph_length: int = 0
    bridges_rejected_by_euclidean_distance: int = 0
    bridges_rejected_by_existing_seam_edge: int = 0
    bridges_rejected_by_skeleton_intersection: int = 0
    bridges_rejected_by_already_connected: int = 0
    bridges_rejected_by_loop: int = 0
    bridges_rejected_by_conflict: int = 0
    bridges_rejected_by_non_mutual: int = 0
    bridges_rejected_by_tangent: int = 0
    bridges_rejected_by_endpoint_consumed: int = 0
    accepted_bridge_reports: tuple[dict, ...] = tuple()
    rejected_bridge_reports: tuple[dict, ...] = tuple()
    unmatched_endpoints: tuple[int, ...] = tuple()
    bridge_length_edges_histogram: dict[int, int] | None = None
    accepted_bridge_edge_indices: tuple[int, ...] = tuple()
    accepted_bridge_edge_keys: tuple[tuple[int, int], ...] = tuple()
    same_component_candidates_considered: int = 0
    same_component_bridges_accepted: int = 0
    same_component_bridges_rejected_by_loop: int = 0
    same_component_bridges_rejected_by_already_connected: int = 0
    unmatched_endpoint_local_candidates: tuple[dict, ...] = tuple()
    same_component_rejected_candidate_reports: tuple[dict, ...] = tuple()
    local_missing_edge_continuity_candidates: tuple[dict, ...] = tuple()
    local_missing_edge_continuity_candidates_total: int = 0
    endpoint_to_skeleton_candidates: tuple[dict, ...] = tuple()
    endpoint_to_skeleton_candidates_total: int = 0
    near_junction_gap_candidates: tuple[dict, ...] = tuple()
    near_junction_gap_candidates_total: int = 0
    max_bridge_edges: int = 6
    max_bridge_euclidean_ratio: float = 0.03
    min_loop_size_to_allow: int = 8
    component_reports: tuple[dict, ...] = tuple()
    r_bridge: int = 6


@dataclass(frozen=True)
class PruningResult:
    pruned_edge_mask: np.ndarray
    removed_edges: frozenset[int]
    iteration_reports: tuple[dict, ...]
    total_iterations: int
    total_branches_pruned: int
    total_edges_removed: int
    protected_leaves_skipped: int
    stale_entries_skipped: int
    l_min: int
    anchor_boundary: bool


@dataclass(frozen=True)
class TopologyPipelineResult:
    final_edge_mask: np.ndarray
    skeleton_result: SkeletonResult
    bridging_result: BridgingResult
    pruning_result: PruningResult
    tau_low: float
    d_max: int
    r_bridge: int
    l_min: int
    anchor_boundary: bool
    max_bridge_euclidean_ratio: float = 0.03
    min_loop_size_to_allow: int = 8


def build_skeleton_subgraph(
    view: SeamGraphView,
    skeleton_edge_mask: np.ndarray,
) -> nx.Graph:
    if skeleton_edge_mask.shape != (view.edge_count,):
        raise ValueError(
            f'skeleton_edge_mask must have shape ({view.edge_count},), got {skeleton_edge_mask.shape}'
        )
    if skeleton_edge_mask.dtype != bool:
        raise ValueError('skeleton_edge_mask must have dtype bool')

    graph = nx.Graph()
    for edge_index in np.flatnonzero(skeleton_edge_mask):
        idx = int(edge_index)
        vi = int(view.unique_edges[idx, 0])
        vj = int(view.unique_edges[idx, 1])
        graph.add_edge(
            vi,
            vj,
            edge_index=idx,
            length=float(view.edge_lengths[idx]),
        )
    return graph


def _validated_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return int(value)


def _validated_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f'{name} must be an integer greater than or equal to 1')
    return int(value)


def _sorted_components(graph: nx.Graph) -> list[frozenset[int]]:
    components = [
        frozenset(int(vertex) for vertex in component)
        for component in nx.connected_components(graph)
    ]
    components.sort(key=lambda component: min(component))
    return components


def _component_id_map(components: list[frozenset[int]]) -> dict[int, int]:
    component_id_of: dict[int, int] = {}
    for component_id, component in enumerate(components):
        for vertex in component:
            component_id_of[int(vertex)] = component_id
    return component_id_of


def _mesh_bbox_diagonal(view: SeamGraphView) -> float:
    if view.vertex_count == 0:
        return 0.0
    coords = np.asarray(view.canonical_vertices, dtype=np.float64).reshape((-1, 3))
    if coords.size == 0:
        return 0.0
    return float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))


def _vertex_distance(view: SeamGraphView, left: int, right: int) -> float:
    coords = np.asarray(view.canonical_vertices, dtype=np.float64)
    return float(np.linalg.norm(coords[int(left)] - coords[int(right)]))


def _bridge_report_base(left: int, right: int, component_id_of: dict[int, int]) -> dict:
    endpoints = [int(min(left, right)), int(max(left, right))]
    return {
        'endpoint_vertex_ids': endpoints,
        'component_ids_before': [
            component_id_of.get(endpoints[0]),
            component_id_of.get(endpoints[1]),
        ],
    }


def _shortest_mesh_edge_path(
    view: SeamGraphView,
    source: int,
    target: int,
    *,
    max_bridge_edges: int,
) -> tuple[tuple[int, ...], tuple[int, ...], float] | None:
    heap: list[tuple[float, int, int, tuple[int, ...], tuple[int, ...]]] = [
        (0.0, 0, int(source), (int(source),), tuple())
    ]
    best: dict[tuple[int, int], float] = {(int(source), 0): 0.0}
    while heap:
        length, edge_count, vertex, path_vertices, path_edges = heapq.heappop(heap)
        if vertex == target and edge_count > 0:
            return path_vertices, path_edges, float(length)
        if edge_count >= max_bridge_edges:
            continue
        for neighbor in sorted(view.vertex_graph.neighbors(vertex)):
            neighbor_index = int(neighbor)
            if neighbor_index in path_vertices:
                continue
            data = view.vertex_graph[vertex][neighbor_index]
            edge_index = int(data['edge_index'])
            next_edge_count = edge_count + 1
            next_length = length + float(data.get('length', view.edge_lengths[edge_index]))
            state = (neighbor_index, next_edge_count)
            if next_length >= best.get(state, float('inf')):
                continue
            best[state] = next_length
            heapq.heappush(
                heap,
                (
                    next_length,
                    next_edge_count,
                    neighbor_index,
                    (*path_vertices, neighbor_index),
                    (*path_edges, edge_index),
                ),
            )
    return None


def _missing_path_reason(view: SeamGraphView, source: int, target: int, max_bridge_edges: int) -> str:
    try:
        hop_count = int(nx.shortest_path_length(view.vertex_graph, source, target))
    except nx.NetworkXNoPath:
        return 'no_path'
    return 'graph_length' if hop_count > max_bridge_edges else 'no_path'


def _endpoint_tangents(
    view: SeamGraphView,
    skeleton_graph: nx.Graph,
    endpoints: list[int],
) -> dict[int, np.ndarray]:
    tangents: dict[int, np.ndarray] = {}
    for endpoint in endpoints:
        neighbors = [int(neighbor) for neighbor in skeleton_graph.neighbors(endpoint)]
        if len(neighbors) != 1:
            continue
        neighbor = neighbors[0]
        vector = _approx_vertex_vector(view, neighbor, endpoint)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            tangents[int(endpoint)] = vector / norm
    return tangents


def _approx_vertex_vector(view: SeamGraphView, source: int, target: int) -> np.ndarray:
    coords = np.asarray(view.canonical_vertices, dtype=np.float64)
    return coords[int(target)] - coords[int(source)]


def _tangent_penalty(
    view: SeamGraphView,
    path_vertices: tuple[int, ...],
    endpoint_tangents: dict[int, np.ndarray],
) -> float:
    if len(path_vertices) < 2:
        return 0.0
    left = int(path_vertices[0])
    right = int(path_vertices[-1])
    penalties: list[float] = []
    if left in endpoint_tangents:
        vector = _approx_vertex_vector(view, left, int(path_vertices[1]))
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            penalties.append(1.0 - float(np.dot(endpoint_tangents[left], vector / norm)))
    if right in endpoint_tangents:
        vector = _approx_vertex_vector(view, right, int(path_vertices[-2]))
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            penalties.append(1.0 - float(np.dot(endpoint_tangents[right], vector / norm)))
    return float(sum(penalties))


def _report_sort_key(report: dict) -> tuple:
    endpoints = report.get('endpoint_vertex_ids')
    if not isinstance(endpoints, list):
        endpoints = [report.get('endpoint_vertex_id', report.get('edge_id', 0)), report.get('target_vertex_id', 0)]
    path_edge_count = report.get('path_edge_count')
    if path_edge_count is None:
        path_edge_count = report.get('graph_distance')
    if path_edge_count is None:
        path_edge_count = 10**9
    euclidean_distance = report.get('euclidean_distance')
    if euclidean_distance is None:
        euclidean_distance = float('inf')
    return (
        int(path_edge_count),
        float(euclidean_distance),
        tuple(int(value) for value in endpoints if isinstance(value, (int, np.integer))),
        int(report.get('edge_id', -1)),
    )


def _bounded_sorted_reports(reports: list[dict]) -> tuple[dict, ...]:
    return tuple(sorted(reports, key=_report_sort_key)[:BRIDGE_MAX_DEBUG_CANDIDATES])


def _path_edge_ids_between(
    view: SeamGraphView,
    left: int,
    right: int,
    *,
    max_bridge_edges: int,
) -> tuple[tuple[int, ...], tuple[int, ...], float] | None:
    return _shortest_mesh_edge_path(view, left, right, max_bridge_edges=max_bridge_edges)


def _candidate_debug_report(
    view: SeamGraphView,
    *,
    left: int,
    right: int,
    component_id_of: dict[int, int],
    skeleton_vertices: set[int],
    seam_edge_indices: set[int],
    reason: str,
    path: tuple[tuple[int, ...], tuple[int, ...], float] | None,
) -> dict:
    path_vertices: tuple[int, ...] = tuple()
    path_edges: tuple[int, ...] = tuple()
    path_length: float | None = None
    if path is not None:
        path_vertices, path_edges, path_length = path
    interior_vertices = path_vertices[1:-1] if len(path_vertices) >= 2 else tuple()
    return {
        'endpoint_vertex_ids': [int(min(left, right)), int(max(left, right))],
        'component_id': component_id_of.get(int(left)),
        'component_ids_before': [
            component_id_of.get(int(min(left, right))),
            component_id_of.get(int(max(left, right))),
        ],
        'path_edge_ids': [int(edge_index) for edge_index in path_edges],
        'path_edge_count': len(path_edges) if path is not None else None,
        'path_length': path_length,
        'euclidean_distance': _vertex_distance(view, left, right),
        'rejection_reason': reason,
        'uses_existing_seam_edge': any(int(edge_index) in seam_edge_indices for edge_index in path_edges),
        'hits_skeleton_interior': any(int(vertex) in skeleton_vertices for vertex in interior_vertices),
        'original_applyable_if_traceable': None,
    }


def _build_local_missing_edge_continuity_reports(
    view: SeamGraphView,
    skeleton_graph: nx.Graph,
    skeleton_mask: np.ndarray,
    component_id_of: dict[int, int],
) -> tuple[tuple[dict, ...], int]:
    reports: list[dict] = []
    for edge_index in range(view.edge_count):
        if bool(skeleton_mask[edge_index]):
            continue
        left = int(view.unique_edges[edge_index, 0])
        right = int(view.unique_edges[edge_index, 1])
        if left not in skeleton_graph or right not in skeleton_graph:
            continue
        left_degree = int(skeleton_graph.degree[left])
        right_degree = int(skeleton_graph.degree[right])
        left_component = component_id_of.get(left)
        right_component = component_id_of.get(right)
        same_component = left_component is not None and left_component == right_component
        loop_size = None
        if same_component and nx.has_path(skeleton_graph, left, right):
            loop_size = int(nx.shortest_path_length(skeleton_graph, left, right)) + 1
        reports.append({
            'edge_id': int(edge_index),
            'vertex_ids_0based': [left, right],
            'seam_degree_left': left_degree,
            'seam_degree_right': right_degree,
            'component_ids': [left_component, right_component],
            'same_component': bool(same_component),
            'would_create_loop': bool(same_component),
            'estimated_loop_size': loop_size,
            'original_applyable_if_traceable': None,
            'why_not_currently_accepted': 'both_vertices_already_in_skeleton',
            'path_edge_count': 1,
            'euclidean_distance': _vertex_distance(view, left, right),
        })
    return _bounded_sorted_reports(reports), len(reports)


def _build_unmatched_endpoint_reports(
    view: SeamGraphView,
    endpoints: list[int],
    unmatched_endpoints: tuple[int, ...],
    component_id_of: dict[int, int],
    rejected_reason_by_pair: dict[tuple[int, int], str],
    generated_candidate_pairs: set[tuple[int, int]],
    *,
    max_bridge_edges: int,
) -> tuple[dict, ...]:
    reports: list[dict] = []
    per_endpoint_limit = max(1, BRIDGE_MAX_DEBUG_CANDIDATES // max(1, len(unmatched_endpoints)))
    for endpoint in unmatched_endpoints:
        local_candidates: list[dict] = []
        for candidate in endpoints:
            if int(candidate) == int(endpoint):
                continue
            try:
                graph_distance = int(nx.shortest_path_length(view.vertex_graph, endpoint, candidate))
            except nx.NetworkXNoPath:
                continue
            if graph_distance > max_bridge_edges:
                continue
            pair_key = tuple(sorted((int(endpoint), int(candidate))))
            local_candidates.append({
                'candidate_endpoint_id': int(candidate),
                'graph_distance': graph_distance,
                'euclidean_distance': _vertex_distance(view, int(endpoint), int(candidate)),
                'same_component': component_id_of.get(int(endpoint)) == component_id_of.get(int(candidate)),
                'component_ids': [
                    component_id_of.get(int(endpoint)),
                    component_id_of.get(int(candidate)),
                ],
                'rejection_reason': rejected_reason_by_pair.get(pair_key),
                'candidate_generated': pair_key in generated_candidate_pairs,
            })
        local_candidates.sort(key=_report_sort_key)
        reports.append({
            'endpoint_vertex_id': int(endpoint),
            'component_id': component_id_of.get(int(endpoint)),
            'nearest_endpoint_candidates': local_candidates[:per_endpoint_limit],
        })
    return tuple(reports[:BRIDGE_MAX_DEBUG_CANDIDATES])


def _build_endpoint_to_skeleton_reports(
    view: SeamGraphView,
    skeleton_graph: nx.Graph,
    endpoints: list[int],
    component_id_of: dict[int, int],
    *,
    max_bridge_edges: int,
) -> tuple[tuple[dict, ...], int]:
    endpoint_set = set(endpoints)
    reports: list[dict] = []
    for endpoint in endpoints:
        for target in sorted(int(vertex) for vertex in skeleton_graph.nodes()):
            if target == endpoint or target in endpoint_set:
                continue
            target_degree = int(skeleton_graph.degree[target])
            if target_degree < 2:
                continue
            path = _path_edge_ids_between(view, endpoint, target, max_bridge_edges=max_bridge_edges)
            if path is None:
                continue
            path_vertices, path_edges, path_length = path
            same_component = component_id_of.get(endpoint) == component_id_of.get(target)
            reports.append({
                'endpoint_vertex_id': int(endpoint),
                'target_skeleton_vertex_id': int(target),
                'target_seam_degree': target_degree,
                'path_vertex_ids': [int(vertex) for vertex in path_vertices],
                'path_edge_ids': [int(edge_index) for edge_index in path_edges],
                'path_edge_count': len(path_edges),
                'path_length': path_length,
                'same_component': bool(same_component),
                'component_ids': [component_id_of.get(endpoint), component_id_of.get(target)],
                'original_applyable_if_traceable': None,
                'reason_not_covered': 'target_is_not_degree_1_endpoint',
                'euclidean_distance': _vertex_distance(view, endpoint, target),
            })
    return _bounded_sorted_reports(reports), len(reports)


def _build_near_junction_gap_reports(
    view: SeamGraphView,
    skeleton_graph: nx.Graph,
    component_id_of: dict[int, int],
    *,
    max_bridge_edges: int,
) -> tuple[tuple[dict, ...], int]:
    junctions = sorted(int(vertex) for vertex, degree in skeleton_graph.degree() if int(degree) >= 3)
    if not junctions:
        return tuple(), 0
    sources = sorted(
        int(vertex)
        for vertex, degree in skeleton_graph.degree()
        if int(degree) == 1 or int(degree) == 2
    )
    reports: list[dict] = []
    for source in sources:
        for junction in junctions:
            if source == junction:
                continue
            path = _path_edge_ids_between(view, source, junction, max_bridge_edges=max_bridge_edges)
            if path is None:
                continue
            path_vertices, path_edges, path_length = path
            reports.append({
                'source_vertex_id': int(source),
                'source_seam_degree': int(skeleton_graph.degree[source]),
                'junction_vertex_id': int(junction),
                'path_vertex_ids': [int(vertex) for vertex in path_vertices],
                'path_edge_ids': [int(edge_index) for edge_index in path_edges],
                'path_edge_count': len(path_edges),
                'path_length': path_length,
                'component_relation': (
                    'same_component'
                    if component_id_of.get(source) == component_id_of.get(junction)
                    else 'different_component'
                ),
                'component_ids': [component_id_of.get(source), component_id_of.get(junction)],
                'original_applyable_if_traceable': None,
                'euclidean_distance': _vertex_distance(view, source, junction),
            })
    return _bounded_sorted_reports(reports), len(reports)


def compute_endpoint_bridging(
    view: SeamGraphView,
    skel_result: SkeletonResult,
    *,
    max_bridge_edges: int = 6,
    max_bridge_euclidean_ratio: float = 0.03,
    min_loop_size_to_allow: int = 8,
) -> BridgingResult:
    max_bridge_edges_value = _validated_nonnegative_int('max_bridge_edges', max_bridge_edges)
    min_loop_size_value = _validated_positive_int('min_loop_size_to_allow', min_loop_size_to_allow)
    max_ratio_value = float(max_bridge_euclidean_ratio)
    if not np.isfinite(max_ratio_value) or max_ratio_value < 0.0:
        raise ValueError('max_bridge_euclidean_ratio must be finite and non-negative')
    if skel_result.skeleton_edge_mask.shape != (view.edge_count,):
        raise ValueError(
            f'skeleton_edge_mask must have shape ({view.edge_count}), '
            f'got {skel_result.skeleton_edge_mask.shape}'
        )
    if skel_result.skeleton_edge_mask.dtype != bool:
        raise ValueError('skeleton_edge_mask must have dtype bool')

    skeleton_mask = skel_result.skeleton_edge_mask
    bridged_mask = skeleton_mask.copy()
    skeleton_graph = build_skeleton_subgraph(view, skeleton_mask)
    components = _sorted_components(skeleton_graph)
    component_id_of = _component_id_map(components)
    component_endpoint_counts: dict[int, int] = {index: 0 for index in range(len(components))}
    endpoints = sorted(int(vertex) for vertex, degree in skeleton_graph.degree() if int(degree) == 1)
    for endpoint in endpoints:
        component_endpoint_counts[component_id_of[endpoint]] += 1

    bbox_diagonal = _mesh_bbox_diagonal(view)
    max_euclidean_distance = max_ratio_value * bbox_diagonal
    skeleton_vertices = set(int(vertex) for vertex in skeleton_graph.nodes())
    seam_edge_indices = set(int(index) for index in np.flatnonzero(skeleton_mask))
    endpoint_tangents = _endpoint_tangents(view, skeleton_graph, endpoints)

    counters = {
        'bridges_rejected_by_no_path': 0,
        'bridges_rejected_by_graph_length': 0,
        'bridges_rejected_by_euclidean_distance': 0,
        'bridges_rejected_by_existing_seam_edge': 0,
        'bridges_rejected_by_skeleton_intersection': 0,
        'bridges_rejected_by_already_connected': 0,
        'bridges_rejected_by_loop': 0,
        'bridges_rejected_by_conflict': 0,
        'bridges_rejected_by_non_mutual': 0,
        'bridges_rejected_by_tangent': 0,
        'bridges_rejected_by_endpoint_consumed': 0,
    }
    rejected_reports: list[dict] = []
    candidate_reports_by_endpoint: dict[int, list[dict]] = {endpoint: [] for endpoint in endpoints}
    endpoint_pairs_considered = 0
    candidate_paths_found = 0
    candidate_paths_valid = 0
    same_component_candidates_considered = 0
    same_component_bridges_rejected_by_loop = 0
    same_component_bridges_rejected_by_already_connected = 0
    same_component_rejected_reports: list[dict] = []
    rejected_reason_by_pair: dict[tuple[int, int], str] = {}
    generated_candidate_pairs: set[tuple[int, int]] = set()

    for left_index, left in enumerate(endpoints):
        for right in endpoints[left_index + 1:]:
            endpoint_pairs_considered += 1
            same_component_pair = component_id_of.get(left) == component_id_of.get(right)
            if same_component_pair:
                same_component_candidates_considered += 1
            euclidean_distance = _vertex_distance(view, left, right)
            base_report = _bridge_report_base(left, right, component_id_of)
            if euclidean_distance > max_euclidean_distance:
                counters['bridges_rejected_by_euclidean_distance'] += 1
                rejected_reason_by_pair[tuple(sorted((left, right)))] = 'euclidean_distance'
                rejected_reports.append({
                    **base_report,
                    'rejection_reason': 'euclidean_distance',
                    'euclidean_distance': euclidean_distance,
                    'max_euclidean_distance': max_euclidean_distance,
                })
                if same_component_pair:
                    same_component_rejected_reports.append(_candidate_debug_report(
                        view,
                        left=left,
                        right=right,
                        component_id_of=component_id_of,
                        skeleton_vertices=skeleton_vertices,
                        seam_edge_indices=seam_edge_indices,
                        reason='euclidean_distance',
                        path=None,
                    ))
                continue

            path = _shortest_mesh_edge_path(view, left, right, max_bridge_edges=max_bridge_edges_value)
            if path is None:
                reason = _missing_path_reason(view, left, right, max_bridge_edges_value)
                counter_key = (
                    'bridges_rejected_by_graph_length'
                    if reason == 'graph_length'
                    else 'bridges_rejected_by_no_path'
                )
                counters[counter_key] += 1
                rejected_reason_by_pair[tuple(sorted((left, right)))] = reason
                rejected_reports.append({**base_report, 'rejection_reason': reason})
                if same_component_pair:
                    same_component_rejected_reports.append(_candidate_debug_report(
                        view,
                        left=left,
                        right=right,
                        component_id_of=component_id_of,
                        skeleton_vertices=skeleton_vertices,
                        seam_edge_indices=seam_edge_indices,
                        reason=reason,
                        path=None,
                    ))
                continue

            candidate_paths_found += 1
            path_vertices, path_edges, path_length = path
            edge_count = len(path_edges)
            path_report = {
                **base_report,
                'path_vertex_ids': list(path_vertices),
                'path_edge_ids': list(path_edges),
                'path_edge_count': edge_count,
                'path_length': path_length,
            }
            if edge_count > max_bridge_edges_value:
                counters['bridges_rejected_by_graph_length'] += 1
                rejected_reason_by_pair[tuple(sorted((left, right)))] = 'graph_length'
                rejected_reports.append({**path_report, 'rejection_reason': 'graph_length'})
                if same_component_pair:
                    same_component_rejected_reports.append(_candidate_debug_report(
                        view,
                        left=left,
                        right=right,
                        component_id_of=component_id_of,
                        skeleton_vertices=skeleton_vertices,
                        seam_edge_indices=seam_edge_indices,
                        reason='graph_length',
                        path=path,
                    ))
                continue
            if any(edge_index in seam_edge_indices for edge_index in path_edges):
                counters['bridges_rejected_by_existing_seam_edge'] += 1
                rejected_reason_by_pair[tuple(sorted((left, right)))] = 'existing_seam_edge'
                rejected_reports.append({**path_report, 'rejection_reason': 'existing_seam_edge'})
                if same_component_pair:
                    same_component_rejected_reports.append(_candidate_debug_report(
                        view,
                        left=left,
                        right=right,
                        component_id_of=component_id_of,
                        skeleton_vertices=skeleton_vertices,
                        seam_edge_indices=seam_edge_indices,
                        reason='existing_seam_edge',
                        path=path,
                    ))
                continue
            interior_vertices = tuple(int(vertex) for vertex in path_vertices[1:-1])
            if any(vertex in skeleton_vertices for vertex in interior_vertices):
                counters['bridges_rejected_by_skeleton_intersection'] += 1
                rejected_reason_by_pair[tuple(sorted((left, right)))] = 'skeleton_intersection'
                rejected_reports.append({**path_report, 'rejection_reason': 'skeleton_intersection'})
                if same_component_pair:
                    same_component_rejected_reports.append(_candidate_debug_report(
                        view,
                        left=left,
                        right=right,
                        component_id_of=component_id_of,
                        skeleton_vertices=skeleton_vertices,
                        seam_edge_indices=seam_edge_indices,
                        reason='skeleton_intersection',
                        path=path,
                    ))
                continue

            same_component = same_component_pair
            loop_size: int | None = None
            if same_component:
                component_id = component_id_of[left]
                if component_endpoint_counts.get(component_id, 0) != 2:
                    counters['bridges_rejected_by_loop'] += 1
                    same_component_bridges_rejected_by_loop += 1
                    rejected_reason_by_pair[tuple(sorted((left, right)))] = 'loop'
                    rejected_reports.append({**path_report, 'rejection_reason': 'loop'})
                    same_component_rejected_reports.append(_candidate_debug_report(
                        view,
                        left=left,
                        right=right,
                        component_id_of=component_id_of,
                        skeleton_vertices=skeleton_vertices,
                        seam_edge_indices=seam_edge_indices,
                        reason='loop',
                        path=path,
                    ))
                    continue
                skeleton_path_edges = nx.shortest_path_length(skeleton_graph, left, right)
                loop_size = int(skeleton_path_edges) + edge_count
                if loop_size < min_loop_size_value:
                    counters['bridges_rejected_by_already_connected'] += 1
                    same_component_bridges_rejected_by_already_connected += 1
                    rejected_reason_by_pair[tuple(sorted((left, right)))] = 'already_connected'
                    rejected_reports.append({
                        **path_report,
                        'rejection_reason': 'already_connected',
                        'loop_size': loop_size,
                    })
                    debug_report = _candidate_debug_report(
                        view,
                        left=left,
                        right=right,
                        component_id_of=component_id_of,
                        skeleton_vertices=skeleton_vertices,
                        seam_edge_indices=seam_edge_indices,
                        reason='already_connected',
                        path=path,
                    )
                    debug_report['loop_size'] = loop_size
                    same_component_rejected_reports.append(debug_report)
                    continue

            tangent_penalty = _tangent_penalty(
                view,
                path_vertices,
                endpoint_tangents,
            )
            weighted_tangent_penalty = BRIDGE_TANGENT_ALIGNMENT_WEIGHT * tangent_penalty
            score = (
                edge_count,
                path_length,
                weighted_tangent_penalty,
                euclidean_distance,
                min(left, right),
                max(left, right),
            )
            report = {
                **path_report,
                'bridge_edge_count': edge_count,
                'bridge_length': path_length,
                'score_tuple': list(score),
                'same_component': bool(same_component),
                'loop_size': loop_size,
                'euclidean_distance': euclidean_distance,
                'interior_vertex_ids': list(interior_vertices),
            }
            candidate_paths_valid += 1
            generated_candidate_pairs.add(tuple(sorted((left, right))))
            candidate_reports_by_endpoint[left].append(report)
            candidate_reports_by_endpoint[right].append(report)

    best_by_endpoint: dict[int, dict] = {}
    candidate_pool: dict[tuple[int, int], dict] = {}
    for endpoint in endpoints:
        ranked = sorted(
            candidate_reports_by_endpoint[endpoint],
            key=lambda report: tuple(report['score_tuple']),
        )[:BRIDGE_MAX_ENDPOINT_CANDIDATES]
        if ranked:
            best_by_endpoint[endpoint] = ranked[0]
        for report in ranked:
            key = tuple(sorted(report['endpoint_vertex_ids']))
            current = candidate_pool.get(key)
            if current is None or tuple(report['score_tuple']) < tuple(current['score_tuple']):
                candidate_pool[key] = report

    eligible: list[dict] = []
    for key, report in sorted(candidate_pool.items(), key=lambda item: tuple(item[1]['score_tuple'])):
        left, right = key
        is_mutual = (
            best_by_endpoint.get(left, {}).get('endpoint_vertex_ids') == [left, right]
            and best_by_endpoint.get(right, {}).get('endpoint_vertex_ids') == [left, right]
        )
        if BRIDGE_REQUIRE_MUTUAL_PAIRING and not is_mutual:
            counters['bridges_rejected_by_non_mutual'] += 1
            rejected_reason_by_pair[tuple(sorted((left, right)))] = 'non_mutual'
            rejected_reports.append({**report, 'rejection_reason': 'non_mutual'})
            continue
        eligible.append(report)

    accepted_reports: list[dict] = []
    consumed_endpoints: set[int] = set()
    reserved_edges: set[int] = set()
    reserved_interior_vertices: set[int] = set()
    added_bridge_edges: set[int] = set()
    accepted_lengths: list[int] = []
    histogram: dict[int, int] = {}
    same_component_bridges_accepted = 0

    for report in sorted(eligible, key=lambda item: tuple(item['score_tuple'])):
        left, right = report['endpoint_vertex_ids']
        if left in consumed_endpoints or right in consumed_endpoints:
            counters['bridges_rejected_by_endpoint_consumed'] += 1
            rejected_reason_by_pair[tuple(sorted((left, right)))] = 'endpoint_consumed'
            rejected_reports.append({**report, 'rejection_reason': 'endpoint_consumed'})
            continue
        path_edges = set(int(edge_index) for edge_index in report['path_edge_ids'])
        interior_vertices = set(int(vertex) for vertex in report['interior_vertex_ids'])
        if path_edges & reserved_edges or interior_vertices & reserved_interior_vertices:
            counters['bridges_rejected_by_conflict'] += 1
            rejected_reason_by_pair[tuple(sorted((left, right)))] = 'conflict'
            rejected_reports.append({**report, 'rejection_reason': 'conflict'})
            continue

        for edge_index in path_edges:
            bridged_mask[edge_index] = True
        consumed_endpoints.update((left, right))
        reserved_edges.update(path_edges)
        reserved_interior_vertices.update(interior_vertices)
        added_bridge_edges.update(path_edges)
        accepted_lengths.append(int(report['bridge_edge_count']))
        histogram[int(report['bridge_edge_count'])] = histogram.get(int(report['bridge_edge_count']), 0) + 1
        accepted_reports.append(dict(report))
        if bool(report.get('same_component')):
            same_component_bridges_accepted += 1

    after_graph = build_skeleton_subgraph(view, bridged_mask)
    endpoints_after = sum(1 for _, degree in after_graph.degree() if int(degree) == 1)
    components_after = sum(1 for _ in nx.connected_components(after_graph)) if after_graph.number_of_edges() else 0
    unmatched = tuple(endpoint for endpoint in endpoints if endpoint not in consumed_endpoints)
    unmatched_endpoint_local_candidates = _build_unmatched_endpoint_reports(
        view,
        endpoints,
        unmatched,
        component_id_of,
        rejected_reason_by_pair,
        generated_candidate_pairs,
        max_bridge_edges=max_bridge_edges_value,
    )
    local_missing_edge_reports, local_missing_edge_total = _build_local_missing_edge_continuity_reports(
        view,
        skeleton_graph,
        skeleton_mask,
        component_id_of,
    )
    endpoint_to_skeleton_reports, endpoint_to_skeleton_total = _build_endpoint_to_skeleton_reports(
        view,
        skeleton_graph,
        endpoints,
        component_id_of,
        max_bridge_edges=max_bridge_edges_value,
    )
    near_junction_reports, near_junction_total = _build_near_junction_gap_reports(
        view,
        skeleton_graph,
        component_id_of,
        max_bridge_edges=max_bridge_edges_value,
    )
    component_reports = tuple(
        {
            'component_id': int(index),
            'skeleton_vertex_count': int(len(component)),
            'skeleton_edge_count': int(skeleton_graph.subgraph(component).number_of_edges()),
            'endpoint_count': int(component_endpoint_counts.get(index, 0)),
        }
        for index, component in enumerate(components)
    )

    return BridgingResult(
        bridged_edge_mask=bridged_mask,
        added_bridge_edges=frozenset(added_bridge_edges),
        endpoints_before=len(endpoints),
        endpoints_after=int(endpoints_after),
        components_before=len(components),
        components_after=int(components_after),
        endpoint_pairs_considered=int(endpoint_pairs_considered),
        candidate_paths_found=int(candidate_paths_found),
        candidate_paths_valid=int(candidate_paths_valid),
        bridges_accepted=len(accepted_reports),
        added_bridge_edges_count=len(added_bridge_edges),
        max_bridge_length_edges=max(accepted_lengths, default=0),
        mean_bridge_length_edges=float(np.mean(accepted_lengths)) if accepted_lengths else 0.0,
        accepted_bridge_reports=tuple(accepted_reports),
        rejected_bridge_reports=tuple(rejected_reports),
        unmatched_endpoints=unmatched,
        bridge_length_edges_histogram=dict(sorted(histogram.items())),
        accepted_bridge_edge_indices=tuple(sorted(int(edge_index) for edge_index in added_bridge_edges)),
        accepted_bridge_edge_keys=tuple(
            (int(view.unique_edges[edge_index, 0]), int(view.unique_edges[edge_index, 1]))
            for edge_index in sorted(added_bridge_edges)
        ),
        same_component_candidates_considered=int(same_component_candidates_considered),
        same_component_bridges_accepted=int(same_component_bridges_accepted),
        same_component_bridges_rejected_by_loop=int(same_component_bridges_rejected_by_loop),
        same_component_bridges_rejected_by_already_connected=int(
            same_component_bridges_rejected_by_already_connected
        ),
        unmatched_endpoint_local_candidates=unmatched_endpoint_local_candidates,
        same_component_rejected_candidate_reports=_bounded_sorted_reports(same_component_rejected_reports),
        local_missing_edge_continuity_candidates=local_missing_edge_reports,
        local_missing_edge_continuity_candidates_total=int(local_missing_edge_total),
        endpoint_to_skeleton_candidates=endpoint_to_skeleton_reports,
        endpoint_to_skeleton_candidates_total=int(endpoint_to_skeleton_total),
        near_junction_gap_candidates=near_junction_reports,
        near_junction_gap_candidates_total=int(near_junction_total),
        max_bridge_edges=max_bridge_edges_value,
        max_bridge_euclidean_ratio=max_ratio_value,
        min_loop_size_to_allow=min_loop_size_value,
        component_reports=component_reports,
        **counters,
    )


def compute_spur_pruning(
    view: SeamGraphView,
    bridging_result: BridgingResult,
    *,
    l_min: int = 4,
    anchor_boundary: bool = True,
    extra_anchor_vertices: frozenset[int] | None = None,
    topology: Any = None,
) -> PruningResult:
    if isinstance(l_min, bool) or not isinstance(l_min, (int, np.integer)) or int(l_min) < 1:
        raise ValueError('l_min must be an integer greater than or equal to 1')
    l_min_value = int(l_min)
    if bridging_result.bridged_edge_mask.shape != (view.edge_count,):
        raise ValueError(
            f'bridged_edge_mask must have shape ({view.edge_count},), '
            f'got {bridging_result.bridged_edge_mask.shape}'
        )
    if bridging_result.bridged_edge_mask.dtype != bool:
        raise ValueError('bridged_edge_mask must have dtype bool')
    if anchor_boundary and topology is None:
        raise ValueError('anchor_boundary=True requires a non-None topology argument')

    normalized_extra_anchors: frozenset[int] | None = None
    if extra_anchor_vertices is not None:
        anchors: set[int] = set()
        for vertex in extra_anchor_vertices:
            if isinstance(vertex, bool) or not isinstance(vertex, (int, np.integer)):
                raise ValueError('extra_anchor_vertices must contain integer vertex indices')
            vertex_index = int(vertex)
            if vertex_index < 0 or vertex_index >= view.vertex_count:
                raise ValueError(
                    f'extra_anchor_vertices contains out-of-range vertex index {vertex_index} '
                    f'for vertex_count={view.vertex_count}'
                )
            anchors.add(vertex_index)
        normalized_extra_anchors = frozenset(anchors)

    A_struct: set[int] = set()
    if anchor_boundary:
        A_struct.update(boundary_vertices_from_topology(topology))
    if normalized_extra_anchors is not None:
        A_struct.update(int(vertex) for vertex in normalized_extra_anchors)

    H = nx.Graph()
    bridged_mask = bridging_result.bridged_edge_mask
    for idx in range(view.edge_count):
        if bridged_mask[idx]:
            u = int(view.unique_edges[idx, 0])
            v = int(view.unique_edges[idx, 1])
            H.add_edge(u, v, edge_index=int(idx))

    deg: dict[int, int] = dict(H.degree())
    pruned_mask = bridged_mask.copy()
    removed_edges_global: set[int] = set()
    iteration_reports: list[dict] = []
    protected_leaves_skipped = 0
    stale_entries_skipped = 0
    total_branches_pruned = 0
    total_edges_removed = 0

    iteration = 0
    while True:
        leaves_examined = 0
        branches_pruned_this_iter = 0
        edges_removed_this_iter = 0
        queue = deque(
            vertex
            for vertex, degree in deg.items()
            if degree == 1 and vertex not in A_struct
        )

        while queue:
            v0 = int(queue.popleft())
            leaves_examined += 1

            if deg.get(v0, 0) != 1:
                stale_entries_skipped += 1
                continue
            if v0 in A_struct:
                protected_leaves_skipped += 1
                continue

            path_vertices: list[int] = [v0]
            path_edge_indices: list[int] = []
            prev: int | None = None
            cur = v0

            while True:
                next_vertex: int | None = None
                next_edge_idx: int | None = None
                for neighbor in H.neighbors(cur):
                    u = int(neighbor)
                    if u == prev:
                        continue
                    next_vertex = u
                    next_edge_idx = int(H[cur][u]['edge_index'])
                    break
                if next_vertex is None or next_edge_idx is None:
                    break

                path_vertices.append(next_vertex)
                path_edge_indices.append(next_edge_idx)
                prev, cur = cur, next_vertex

                if deg[cur] >= 3:
                    break
                if deg[cur] == 1:
                    break
                if cur == v0:
                    break

            branch_length = len(path_edge_indices)
            if branch_length >= l_min_value:
                continue

            for edge_index in path_edge_indices:
                pruned_mask[edge_index] = False
                removed_edges_global.add(edge_index)
            for i in range(len(path_vertices) - 1):
                u = path_vertices[i]
                w = path_vertices[i + 1]
                if H.has_edge(u, w):
                    H.remove_edge(u, w)
                    deg[u] -= 1
                    deg[w] -= 1
            branches_pruned_this_iter += 1
            edges_removed_this_iter += branch_length

        iteration_reports.append({
            'iteration': iteration,
            'leaves_examined': leaves_examined,
            'branches_pruned': branches_pruned_this_iter,
            'edges_removed': edges_removed_this_iter,
        })
        total_branches_pruned += branches_pruned_this_iter
        total_edges_removed += edges_removed_this_iter
        iteration += 1

        if branches_pruned_this_iter == 0:
            break

    return PruningResult(
        pruned_edge_mask=pruned_mask,
        removed_edges=frozenset(removed_edges_global),
        iteration_reports=tuple(iteration_reports),
        total_iterations=iteration,
        total_branches_pruned=total_branches_pruned,
        total_edges_removed=total_edges_removed,
        protected_leaves_skipped=protected_leaves_skipped,
        stale_entries_skipped=stale_entries_skipped,
        l_min=l_min_value,
        anchor_boundary=bool(anchor_boundary),
    )


def diagnose_pruning_application(
    view: SeamGraphView,
    bridging_result: BridgingResult,
    *,
    l_min: int = 4,
    anchor_boundary: bool = True,
    extra_anchor_vertices: frozenset[int] | None = None,
    topology: Any = None,
    diagnostics_threshold: float = 0.5,
) -> tuple[PruningResult, SeamMaskDiagnostics, SeamMaskDiagnostics]:
    before_probs = np.where(bridging_result.bridged_edge_mask, 1.0, 0.0).astype(np.float64, copy=False)
    before = compute_seam_mask_diagnostics(view, before_probs, threshold=diagnostics_threshold)
    pruning = compute_spur_pruning(
        view,
        bridging_result,
        l_min=l_min,
        anchor_boundary=anchor_boundary,
        extra_anchor_vertices=extra_anchor_vertices,
        topology=topology,
    )
    after_probs = np.where(pruning.pruned_edge_mask, 1.0, 0.0).astype(np.float64, copy=False)
    after = compute_seam_mask_diagnostics(view, after_probs, threshold=diagnostics_threshold)
    return pruning, before, after


def apply_topology_pipeline(
    view: SeamGraphView,
    probabilities: np.ndarray,
    *,
    tau_low: float = 0.30,
    d_max: int = 3,
    r_bridge: int = 6,
    l_min: int = 4,
    max_bridge_euclidean_ratio: float = 0.03,
    min_loop_size_to_allow: int = 8,
    anchor_boundary: bool = True,
    extra_anchor_vertices: frozenset[int] | None = None,
    topology: Any = None,
) -> TopologyPipelineResult:
    if anchor_boundary and topology is None:
        raise ValueError('anchor_boundary=True requires a non-None topology argument')

    skel = compute_topology_preserving_skeleton(
        view,
        probabilities,
        tau_low=tau_low,
        d_max=d_max,
        anchor_boundary=anchor_boundary,
        extra_anchor_vertices=extra_anchor_vertices,
        topology=topology,
    )
    bridge = compute_endpoint_bridging(
        view,
        skel,
        max_bridge_edges=r_bridge,
        max_bridge_euclidean_ratio=max_bridge_euclidean_ratio,
        min_loop_size_to_allow=min_loop_size_to_allow,
    )
    prune = compute_spur_pruning(
        view,
        bridge,
        l_min=l_min,
        anchor_boundary=anchor_boundary,
        extra_anchor_vertices=extra_anchor_vertices,
        topology=topology,
    )
    return TopologyPipelineResult(
        final_edge_mask=prune.pruned_edge_mask,
        skeleton_result=skel,
        bridging_result=bridge,
        pruning_result=prune,
        tau_low=float(tau_low),
        d_max=int(d_max),
        r_bridge=int(r_bridge),
        l_min=int(l_min),
        anchor_boundary=bool(anchor_boundary),
        max_bridge_euclidean_ratio=float(max_bridge_euclidean_ratio),
        min_loop_size_to_allow=int(min_loop_size_to_allow),
    )


def topology_pipeline_result_to_json_dict(
    result: TopologyPipelineResult,
) -> dict:
    skeleton = result.skeleton_result
    bridging = result.bridging_result
    pruning = result.pruning_result
    accepted_bridge_edge_indices = tuple(sorted(int(index) for index in bridging.accepted_bridge_edge_indices))
    survived_bridge_edges = tuple(
        edge_index
        for edge_index in accepted_bridge_edge_indices
        if bool(result.final_edge_mask[edge_index])
    )
    removed_bridge_edges = tuple(
        edge_index
        for edge_index in accepted_bridge_edge_indices
        if not bool(result.final_edge_mask[edge_index])
    )
    accepted_bridge_edge_keys = {
        int(edge_index): [int(edge_key[0]), int(edge_key[1])]
        for edge_index, edge_key in zip(
            bridging.accepted_bridge_edge_indices,
            bridging.accepted_bridge_edge_keys,
        )
    }
    bridge_edge_ids_final_presence = [
        {
            'edge_id': int(edge_index),
            'vertex_ids_0based': accepted_bridge_edge_keys[int(edge_index)],
            'in_after_stage_b': bool(bridging.bridged_edge_mask[edge_index]),
            'in_after_stage_c': bool(result.final_edge_mask[edge_index]),
            'in_output_seam_edge_indices': bool(result.final_edge_mask[edge_index]),
            'in_output_seam_edges': bool(result.final_edge_mask[edge_index]),
            'original_blender_edge_if_traceable': None,
            'applied_by_blender_if_traceable': None,
        }
        for edge_index in accepted_bridge_edge_indices
    ]
    payload = {
        'bridging': {
            'accepted_bridge_reports': [dict(report) for report in bridging.accepted_bridge_reports],
            'accepted_bridge_edge_indices': [int(index) for index in accepted_bridge_edge_indices],
            'accepted_bridge_edge_keys': [
                [int(edge_key[0]), int(edge_key[1])]
                for edge_key in bridging.accepted_bridge_edge_keys
            ],
            'accepted_bridge_edges_ignored_by_blender_if_traceable': None,
            'accepted_bridge_edges_non_original_if_traceable': None,
            'accepted_bridge_edges_removed_by_stage_c': int(len(removed_bridge_edges)),
            'accepted_bridge_edges_survived_to_final': int(len(survived_bridge_edges)),
            'added_bridge_edges': int(bridging.added_bridge_edges_count),
            'bridge_length_edges_histogram': {
                str(key): int(value)
                for key, value in sorted((bridging.bridge_length_edges_histogram or {}).items())
            },
            'bridge_edge_ids_final_presence': bridge_edge_ids_final_presence,
            'bridges_accepted': int(bridging.bridges_accepted),
            'bridges_rejected_by_already_connected': int(bridging.bridges_rejected_by_already_connected),
            'bridges_rejected_by_conflict': int(bridging.bridges_rejected_by_conflict),
            'bridges_rejected_by_endpoint_consumed': int(bridging.bridges_rejected_by_endpoint_consumed),
            'bridges_rejected_by_euclidean_distance': int(bridging.bridges_rejected_by_euclidean_distance),
            'bridges_rejected_by_existing_seam_edge': int(bridging.bridges_rejected_by_existing_seam_edge),
            'bridges_rejected_by_graph_length': int(bridging.bridges_rejected_by_graph_length),
            'bridges_rejected_by_loop': int(bridging.bridges_rejected_by_loop),
            'bridges_rejected_by_no_path': int(bridging.bridges_rejected_by_no_path),
            'bridges_rejected_by_non_mutual': int(bridging.bridges_rejected_by_non_mutual),
            'bridges_rejected_by_skeleton_intersection': int(bridging.bridges_rejected_by_skeleton_intersection),
            'bridges_rejected_by_tangent': int(bridging.bridges_rejected_by_tangent),
            'candidate_paths_found': int(bridging.candidate_paths_found),
            'candidate_paths_valid': int(bridging.candidate_paths_valid),
            'component_reports': [dict(report) for report in bridging.component_reports],
            'components_after': int(bridging.components_after),
            'components_before': int(bridging.components_before),
            'endpoint_pairs_considered': int(bridging.endpoint_pairs_considered),
            'endpoints_after': int(bridging.endpoints_after),
            'endpoints_before': int(bridging.endpoints_before),
            'final_output_contains_accepted_bridge_edges': bool(len(survived_bridge_edges) > 0),
            'max_bridge_length_edges': int(bridging.max_bridge_length_edges),
            'mean_bridge_length_edges': float(bridging.mean_bridge_length_edges),
            'parameters': {
                'max_bridge_edges': int(bridging.max_bridge_edges),
                'max_bridge_euclidean_ratio': float(bridging.max_bridge_euclidean_ratio),
                'min_loop_size_to_allow': int(bridging.min_loop_size_to_allow),
            },
            'rejected_bridge_reports': [dict(report) for report in bridging.rejected_bridge_reports],
            'same_component_bridges_accepted': int(bridging.same_component_bridges_accepted),
            'same_component_bridges_rejected_by_already_connected': int(
                bridging.same_component_bridges_rejected_by_already_connected
            ),
            'same_component_bridges_rejected_by_loop': int(bridging.same_component_bridges_rejected_by_loop),
            'same_component_candidates_considered': int(bridging.same_component_candidates_considered),
            'same_component_rejected_candidate_reports': [
                dict(report) for report in bridging.same_component_rejected_candidate_reports
            ],
            'seam_edge_count_after_stage_b': int(np.count_nonzero(bridging.bridged_edge_mask)),
            'seam_edge_count_after_stage_c': int(np.count_nonzero(result.final_edge_mask)),
            'seam_edge_count_before_stage_b': int(np.count_nonzero(skeleton.skeleton_edge_mask)),
            'unmatched_endpoint_local_candidates': [
                dict(report) for report in bridging.unmatched_endpoint_local_candidates
            ],
            'local_missing_edge_continuity_candidates': [
                dict(report) for report in bridging.local_missing_edge_continuity_candidates
            ],
            'local_missing_edge_continuity_candidates_total': int(
                bridging.local_missing_edge_continuity_candidates_total
            ),
            'endpoint_to_skeleton_candidates': [
                dict(report) for report in bridging.endpoint_to_skeleton_candidates
            ],
            'endpoint_to_skeleton_candidates_total': int(bridging.endpoint_to_skeleton_candidates_total),
            'near_junction_gap_candidates': [
                dict(report) for report in bridging.near_junction_gap_candidates
            ],
            'near_junction_gap_candidates_total': int(bridging.near_junction_gap_candidates_total),
            'unmatched_endpoints': [int(vertex) for vertex in bridging.unmatched_endpoints],
        },
        'final_edge_count': int(np.count_nonzero(result.final_edge_mask)),
        'parameters': {
            'anchor_boundary': bool(result.anchor_boundary),
            'd_max': int(result.d_max),
            'l_min': int(result.l_min),
            'max_bridge_euclidean_ratio': float(result.max_bridge_euclidean_ratio),
            'min_loop_size_to_allow': int(result.min_loop_size_to_allow),
            'r_bridge': int(result.r_bridge),
            'tau_low': float(result.tau_low),
        },
        'pruning': {
            'anchor_boundary': bool(pruning.anchor_boundary),
            'iteration_reports': [dict(report) for report in pruning.iteration_reports],
            'l_min': int(pruning.l_min),
            'protected_leaves_skipped': int(pruning.protected_leaves_skipped),
            'removed_edges_count': int(len(pruning.removed_edges)),
            'stale_entries_skipped': int(pruning.stale_entries_skipped),
            'total_branches_pruned': int(pruning.total_branches_pruned),
            'total_edges_removed': int(pruning.total_edges_removed),
            'total_iterations': int(pruning.total_iterations),
        },
        'skeleton': {
            'anchor_boundary': bool(skeleton.anchor_boundary),
            'anchor_vertex_count': int(len(skeleton.anchor_vertices)),
            'd_max': int(skeleton.d_max),
            'initial_candidate_count': int(len(skeleton.initial_candidate_vertices)),
            'iterations_performed': int(skeleton.iterations_performed),
            'refused_by_anchor': int(skeleton.refused_by_anchor),
            'refused_by_distance_test': int(skeleton.refused_by_distance_test),
            'refused_by_simple_test': int(skeleton.refused_by_simple_test),
            'removals_committed': int(skeleton.removals_committed),
            'skeleton_vertex_count': int(len(skeleton.skeleton_vertices)),
            'tau_low': float(skeleton.tau_low),
        },
    }
    return {key: payload[key] for key in sorted(payload)}


def _validated_probability_vector(view: SeamGraphView, probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (view.edge_count,):
        raise ValueError(f'probabilities must have shape ({view.edge_count},), got {probs.shape}')
    if not np.isfinite(probs).all():
        raise ValueError('probabilities must be finite')
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError('probabilities must lie in [0.0, 1.0]')
    return probs


def _validated_probability_threshold(name: str, value: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f'{name} must be finite and lie in [0.0, 1.0]')
    return threshold


def _passes_simple_vertex_test(
    adjacency: nx.classes.coreviews.AdjacencyView,
    vertex: int,
    in_C: np.ndarray,
    *,
    depth_bound: int,
) -> bool:
    candidate_neighbors = [
        int(neighbor)
        for neighbor in adjacency[vertex]
        if in_C[int(neighbor)]
    ]
    if len(candidate_neighbors) <= 1:
        return True

    start = candidate_neighbors[0]
    remaining_targets = set(candidate_neighbors[1:])
    visited = {vertex, start}
    queue: deque[tuple[int, int]] = deque([(start, 0)])

    while queue:
        current, depth = queue.popleft()
        if current in remaining_targets:
            remaining_targets.remove(current)
            if not remaining_targets:
                return True
        if depth >= depth_bound:
            continue
        for neighbor in adjacency[current]:
            neighbor_index = int(neighbor)
            if neighbor_index in visited or not in_C[neighbor_index]:
                continue
            visited.add(neighbor_index)
            queue.append((neighbor_index, depth + 1))

    return False


def _bounded_distance_to_candidate_set(
    adjacency: nx.classes.coreviews.AdjacencyView,
    source: int,
    in_C: np.ndarray,
    *,
    max_distance: int,
    excluded_candidate: int | None,
) -> int | None:
    visited = {source}
    queue: deque[tuple[int, int]] = deque([(source, 0)])

    while queue:
        current, distance = queue.popleft()
        if current != source and in_C[current] and current != excluded_candidate:
            return distance
        if distance >= max_distance:
            continue
        for neighbor in adjacency[current]:
            neighbor_index = int(neighbor)
            if neighbor_index in visited:
                continue
            if in_C[neighbor_index] and neighbor_index != excluded_candidate:
                return distance + 1
            visited.add(neighbor_index)
            queue.append((neighbor_index, distance + 1))

    return None


def _deleted_vertices_within_radius(
    adjacency: nx.classes.coreviews.AdjacencyView,
    source: int,
    deleted_vertices: set[int],
    *,
    radius: int,
) -> set[int]:
    if not deleted_vertices:
        return set()

    found: set[int] = set()
    visited = {source}
    queue: deque[tuple[int, int]] = deque([(source, 0)])
    while queue:
        current, distance = queue.popleft()
        if current in deleted_vertices:
            found.add(current)
        if distance >= radius:
            continue
        for neighbor in adjacency[current]:
            neighbor_index = int(neighbor)
            if neighbor_index in visited:
                continue
            visited.add(neighbor_index)
            queue.append((neighbor_index, distance + 1))
    return found
