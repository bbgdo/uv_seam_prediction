from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Any

import networkx as nx
import numpy as np

from preprocessing.topology import CanonicalTopology


_COMPONENT_SIZE_BUCKETS: tuple[str, ...] = (
    '1',
    '2',
    '3-5',
    '6-10',
    '11-20',
    '21-50',
    '51-100',
    '101-500',
    '501+',
)
_BRANCH_LENGTH_BUCKETS: tuple[str, ...] = (
    '1',
    '2',
    '3',
    '4-5',
    '6-10',
    '11-20',
    '21+',
)
_GAP_DISTANCE_BUCKETS: tuple[str, ...] = (
    '0',
    '1',
    '2',
    '3',
    '4-5',
    '6-10',
    '11+',
)


@dataclass(frozen=True)
class SeamGraphView:
    vertex_count: int
    edge_count: int
    unique_edges: np.ndarray
    edge_lengths: np.ndarray
    vertex_to_edges: tuple[tuple[int, ...], ...]
    vertex_graph: nx.Graph


def build_seam_graph_view(
    topology: CanonicalTopology,
    unique_edges: np.ndarray,
) -> SeamGraphView:
    expected_edges = np.asarray(topology.canonical_edges, dtype=np.int64).reshape((-1, 2))
    observed_edges = np.asarray(unique_edges, dtype=np.int64)
    if observed_edges.shape != expected_edges.shape:
        raise ValueError(
            'unique_edges shape mismatch: '
            f'expected {expected_edges.shape} from topology.canonical_edges, got {observed_edges.shape}'
        )
    if not np.array_equal(observed_edges, expected_edges):
        mismatch_rows = np.flatnonzero(np.any(observed_edges != expected_edges, axis=1))
        mismatch_index = int(mismatch_rows[0]) if mismatch_rows.size else -1
        raise ValueError(
            'unique_edges must match topology.canonical_edges exactly; '
            f'first mismatch at edge index {mismatch_index}'
        )

    canonical_vertices = np.asarray(topology.canonical_vertices, dtype=np.float64).reshape((-1, 3))
    vertex_count = int(len(canonical_vertices))
    edge_count = int(len(expected_edges))
    edge_lengths = np.zeros(edge_count, dtype=np.float64)
    vertex_to_edges_lists: list[list[int]] = [[] for _ in range(vertex_count)]
    vertex_graph = nx.Graph()
    vertex_graph.add_nodes_from(range(vertex_count))

    for edge_index, edge in enumerate(expected_edges):
        vi = int(edge[0])
        vj = int(edge[1])
        if vi < 0 or vj < 0 or vi >= vertex_count or vj >= vertex_count:
            raise ValueError(
                f'edge index {edge_index} references out-of-range vertex ids {(vi, vj)} '
                f'for vertex_count={vertex_count}'
            )
        if vi >= vj:
            raise ValueError(f'canonical edge at index {edge_index} is not ordered as vi < vj: {(vi, vj)}')
        length = float(np.linalg.norm(canonical_vertices[vi] - canonical_vertices[vj]))
        edge_lengths[edge_index] = length
        vertex_to_edges_lists[vi].append(edge_index)
        vertex_to_edges_lists[vj].append(edge_index)
        vertex_graph.add_edge(vi, vj, edge_index=edge_index, length=length)

    vertex_to_edges = tuple(tuple(indices) for indices in vertex_to_edges_lists)
    return SeamGraphView(
        vertex_count=vertex_count,
        edge_count=edge_count,
        unique_edges=expected_edges.copy(),
        edge_lengths=edge_lengths,
        vertex_to_edges=vertex_to_edges,
        vertex_graph=vertex_graph,
    )


@dataclass(frozen=True)
class SeamMaskDiagnostics:
    threshold: float
    seam_edge_count: int
    seam_vertex_count: int
    component_count: int
    component_size_histogram: dict[str, int]
    vertex_degree_histogram: dict[int, int]
    junction_count: int
    isolated_edge_count: int
    branch_length_histogram: dict[str, int]
    branch_count: int
    gap_distance_histogram: dict[str, int]
    thick_band_edge_count: int
    mean_probability_in_seam: float
    mean_probability_outside_seam: float


def compute_seam_mask_diagnostics(
    view: SeamGraphView,
    probabilities: np.ndarray,
    threshold: float,
) -> SeamMaskDiagnostics:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (view.edge_count,):
        raise ValueError(
            f'probabilities must have shape ({view.edge_count},), got {probs.shape}'
        )
    if not np.isfinite(probs).all():
        raise ValueError('probabilities must be finite')
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError('probabilities must lie in [0.0, 1.0]')

    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or threshold_value < 0.0 or threshold_value > 1.0:
        raise ValueError(f'threshold must be finite and lie in [0.0, 1.0], got {threshold!r}')

    if view.vertex_count == 0:
        return SeamMaskDiagnostics(
            threshold=threshold_value,
            seam_edge_count=0,
            seam_vertex_count=0,
            component_count=0,
            component_size_histogram={},
            vertex_degree_histogram={},
            junction_count=0,
            isolated_edge_count=0,
            branch_length_histogram={},
            branch_count=0,
            gap_distance_histogram={},
            thick_band_edge_count=0,
            mean_probability_in_seam=0.0,
            mean_probability_outside_seam=0.0,
        )

    seam_mask = probs >= threshold_value
    seam_edge_count = int(np.count_nonzero(seam_mask))
    if seam_edge_count == 0:
        return SeamMaskDiagnostics(
            threshold=threshold_value,
            seam_edge_count=0,
            seam_vertex_count=0,
            component_count=0,
            component_size_histogram={},
            vertex_degree_histogram={},
            junction_count=0,
            isolated_edge_count=0,
            branch_length_histogram={},
            branch_count=0,
            gap_distance_histogram={},
            thick_band_edge_count=0,
            mean_probability_in_seam=0.0,
            mean_probability_outside_seam=float(probs.mean()) if probs.size else 0.0,
        )

    seam_edges = [
        (int(edge[0]), int(edge[1]))
        for edge in view.unique_edges[seam_mask]
    ]
    seam_subgraph = view.vertex_graph.edge_subgraph(seam_edges).copy()
    seam_vertex_count = int(seam_subgraph.number_of_nodes())
    component_nodes = [set(component) for component in nx.connected_components(seam_subgraph)]
    component_edge_counts: list[int] = []
    component_representatives: list[int] = []
    component_size_histogram: dict[str, int] = {}
    isolated_edge_count = 0

    for nodes in component_nodes:
        component_graph = seam_subgraph.subgraph(nodes)
        edge_count = int(component_graph.number_of_edges())
        component_edge_counts.append(edge_count)
        edge_indices = sorted(int(data['edge_index']) for _, _, data in component_graph.edges(data=True))
        component_representatives.append(edge_indices[0] if edge_indices else -1)
        _increment_bucket(component_size_histogram, _bucket_component_size(edge_count))
        if edge_count == 1:
            isolated_edge_count += 1

    degree_histogram: dict[int, int] = {}
    junction_count = 0
    thick_band_edge_count = 0
    for _, degree in seam_subgraph.degree():
        degree_value = int(degree)
        degree_histogram[degree_value] = degree_histogram.get(degree_value, 0) + 1
        if degree_value >= 3:
            junction_count += 1

    for u, v in seam_subgraph.edges():
        if seam_subgraph.degree[u] >= 3 and seam_subgraph.degree[v] >= 3:
            thick_band_edge_count += 1

    branch_length_histogram: dict[str, int] = {}
    branch_count = 0
    for nodes in component_nodes:
        component_graph = seam_subgraph.subgraph(nodes).copy()
        leaf_nodes = sorted(node for node, degree in component_graph.degree() if int(degree) == 1)
        branch_count += len(leaf_nodes)
        for leaf in leaf_nodes:
            branch_length = _branch_length_from_leaf(component_graph, int(leaf))
            _increment_bucket(branch_length_histogram, _bucket_branch_length(branch_length))

    gap_distance_histogram: dict[str, int] = {}
    if len(component_nodes) > 1:
        ranked_components = sorted(
            zip(component_nodes, component_edge_counts, component_representatives),
            key=lambda item: (-item[1], item[2]),
        )
        selected_components = [nodes for nodes, _, _ in ranked_components[:min(20, len(ranked_components))]]
        for left_index in range(len(selected_components)):
            for right_index in range(left_index + 1, len(selected_components)):
                distance = _minimum_vertex_set_distance(
                    view.vertex_graph,
                    selected_components[left_index],
                    selected_components[right_index],
                )
                if distance is None:
                    continue
                _increment_bucket(gap_distance_histogram, _bucket_gap_distance(distance))

    seam_probabilities = probs[seam_mask]
    outside_probabilities = probs[~seam_mask]
    return SeamMaskDiagnostics(
        threshold=threshold_value,
        seam_edge_count=seam_edge_count,
        seam_vertex_count=seam_vertex_count,
        component_count=len(component_nodes),
        component_size_histogram=_ordered_bucket_dict(component_size_histogram, _COMPONENT_SIZE_BUCKETS),
        vertex_degree_histogram=dict(sorted(degree_histogram.items())),
        junction_count=junction_count,
        isolated_edge_count=isolated_edge_count,
        branch_length_histogram=_ordered_bucket_dict(branch_length_histogram, _BRANCH_LENGTH_BUCKETS),
        branch_count=branch_count,
        gap_distance_histogram=_ordered_bucket_dict(gap_distance_histogram, _GAP_DISTANCE_BUCKETS),
        thick_band_edge_count=thick_band_edge_count,
        mean_probability_in_seam=float(seam_probabilities.mean()) if seam_probabilities.size else 0.0,
        mean_probability_outside_seam=float(outside_probabilities.mean()) if outside_probabilities.size else 0.0,
    )


def diagnostics_to_json_dict(d: SeamMaskDiagnostics) -> dict:
    payload = {
        'branch_count': int(d.branch_count),
        'branch_length_histogram': _ordered_string_dict(d.branch_length_histogram, _BRANCH_LENGTH_BUCKETS),
        'component_count': int(d.component_count),
        'component_size_histogram': _ordered_string_dict(d.component_size_histogram, _COMPONENT_SIZE_BUCKETS),
        'gap_distance_histogram': _ordered_string_dict(d.gap_distance_histogram, _GAP_DISTANCE_BUCKETS),
        'isolated_edge_count': int(d.isolated_edge_count),
        'junction_count': int(d.junction_count),
        'mean_probability_in_seam': float(d.mean_probability_in_seam),
        'mean_probability_outside_seam': float(d.mean_probability_outside_seam),
        'seam_edge_count': int(d.seam_edge_count),
        'seam_vertex_count': int(d.seam_vertex_count),
        'thick_band_edge_count': int(d.thick_band_edge_count),
        'threshold': float(d.threshold),
        'vertex_degree_histogram': {str(key): int(value) for key, value in sorted(d.vertex_degree_histogram.items())},
    }
    return {key: payload[key] for key in sorted(payload)}


def _branch_length_from_leaf(graph: nx.Graph, leaf: int) -> int:
    previous: int | None = None
    current = leaf
    length = 0

    while True:
        neighbors = [int(node) for node in graph.neighbors(current) if int(node) != previous]
        if not neighbors:
            return length
        next_node = neighbors[0]
        length += 1
        if int(graph.degree[next_node]) != 2:
            return length
        previous = current
        current = next_node


def _minimum_vertex_set_distance(
    graph: nx.Graph,
    source_vertices: set[int],
    target_vertices: set[int],
) -> int | None:
    if source_vertices & target_vertices:
        return 0

    queue: deque[tuple[int, int]] = deque((vertex, 0) for vertex in sorted(source_vertices))
    visited = set(source_vertices)
    while queue:
        vertex, distance = queue.popleft()
        for neighbor in graph.neighbors(vertex):
            neighbor_value = int(neighbor)
            if neighbor_value in visited:
                continue
            if neighbor_value in target_vertices:
                return distance + 1
            visited.add(neighbor_value)
            queue.append((neighbor_value, distance + 1))
    return None


def _bucket_component_size(size: int) -> str:
    if size == 1:
        return '1'
    if size == 2:
        return '2'
    if size <= 5:
        return '3-5'
    if size <= 10:
        return '6-10'
    if size <= 20:
        return '11-20'
    if size <= 50:
        return '21-50'
    if size <= 100:
        return '51-100'
    if size <= 500:
        return '101-500'
    return '501+'


def _bucket_branch_length(length: int) -> str:
    if length == 1:
        return '1'
    if length == 2:
        return '2'
    if length == 3:
        return '3'
    if length <= 5:
        return '4-5'
    if length <= 10:
        return '6-10'
    if length <= 20:
        return '11-20'
    return '21+'


def _bucket_gap_distance(distance: int) -> str:
    if distance == 0:
        return '0'
    if distance == 1:
        return '1'
    if distance == 2:
        return '2'
    if distance == 3:
        return '3'
    if distance <= 5:
        return '4-5'
    if distance <= 10:
        return '6-10'
    return '11+'


def _increment_bucket(histogram: dict[str, int], bucket: str) -> None:
    histogram[bucket] = histogram.get(bucket, 0) + 1


def _ordered_bucket_dict(histogram: dict[str, int], bucket_order: tuple[str, ...]) -> dict[str, int]:
    return {
        bucket: int(histogram[bucket])
        for bucket in bucket_order
        if histogram.get(bucket, 0) > 0
    }


def _ordered_string_dict(histogram: dict[str, int], bucket_order: tuple[str, ...]) -> dict[str, int]:
    return {
        bucket: int(histogram[bucket])
        for bucket in bucket_order
        if histogram.get(bucket, 0) > 0
    }


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
    steiner_added_edges: frozenset[int]
    component_reports: tuple[dict, ...]
    component_count: int
    terminals_total: int
    terminals_dropped_no_component: int
    steiner_calls: int
    steiner_edges_added_total: int
    tau_high: float
    r_bridge: int
    epsilon: float


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
    tau_high: float
    d_max: int
    r_bridge: int
    l_min: int
    epsilon: float
    anchor_boundary: bool


def build_skeleton_subgraph(
    view: SeamGraphView,
    skeleton_edge_mask: np.ndarray,
) -> nx.Graph:
    """
    Return the graph induced by skeleton edges, preserving canonical edge metadata.
    """
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


def _bounded_search_graph(
    view: SeamGraphView,
    seed_vertices: frozenset[int],
    r_bridge: int,
    probabilities: np.ndarray,
    skeleton_edge_mask: np.ndarray,
    epsilon: float,
) -> nx.Graph:
    """
    Build the mesh subgraph within r_bridge BFS hops of seed_vertices.
    """
    if not seed_vertices:
        return nx.Graph()

    visited: set[int] = {int(vertex) for vertex in seed_vertices}
    queue: deque[tuple[int, int]] = deque((int(vertex), 0) for vertex in sorted(seed_vertices))
    while queue:
        vertex, depth = queue.popleft()
        if depth >= r_bridge:
            continue
        for neighbor in view.vertex_graph.neighbors(vertex):
            neighbor_index = int(neighbor)
            if neighbor_index in visited:
                continue
            visited.add(neighbor_index)
            queue.append((neighbor_index, depth + 1))

    graph = nx.Graph()
    graph.add_nodes_from(sorted(visited))
    for u, v, data in view.vertex_graph.subgraph(visited).edges(data=True):
        idx = int(data['edge_index'])
        if skeleton_edge_mask[idx]:
            weight = 0.0
        else:
            weight = float(max(0.0, -np.log(max(float(probabilities[idx]), float(epsilon)))))
        graph.add_edge(
            int(u),
            int(v),
            edge_index=idx,
            length=float(data.get('length', view.edge_lengths[idx])),
            weight=weight,
        )
    return graph


def compute_steiner_bridging(
    view: SeamGraphView,
    probabilities: np.ndarray,
    skel_result: SkeletonResult,
    *,
    tau_high: float = 0.70,
    r_bridge: int = 6,
    epsilon: float = 1e-3,
    anchor_boundary: bool = True,
    extra_anchor_vertices: frozenset[int] | None = None,
    topology: Any = None,
) -> BridgingResult:
    probs = _validated_probability_vector(view, probabilities)
    tau_high_value = _validated_probability_threshold('tau_high', tau_high)
    if isinstance(r_bridge, bool) or not isinstance(r_bridge, (int, np.integer)) or int(r_bridge) < 0:
        raise ValueError('r_bridge must be a non-negative integer')
    r_bridge_value = int(r_bridge)
    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0 or epsilon_value > 1.0:
        raise ValueError('epsilon must be finite and lie in (0.0, 1.0]')
    if anchor_boundary and topology is None:
        raise ValueError('anchor_boundary=True requires a non-None topology argument')
    if skel_result.skeleton_edge_mask.shape != (view.edge_count,):
        raise ValueError(
            f'skeleton_edge_mask must have shape ({view.edge_count},), '
            f'got {skel_result.skeleton_edge_mask.shape}'
        )
    if skel_result.skeleton_edge_mask.dtype != bool:
        raise ValueError('skeleton_edge_mask must have dtype bool')

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

    vertex_scores = skel_result.vertex_scores
    G_skel = build_skeleton_subgraph(view, skel_result.skeleton_edge_mask)
    components = [frozenset(int(vertex) for vertex in component) for component in nx.connected_components(G_skel)]
    component_id_of: dict[int, int] = {}
    for component_id, component in enumerate(components):
        for vertex in component:
            component_id_of[int(vertex)] = component_id

    T_structural_raw: set[int] = set()
    if anchor_boundary:
        T_structural_raw.update(boundary_vertices_from_topology(topology))
    if normalized_extra_anchors is not None:
        T_structural_raw.update(int(vertex) for vertex in normalized_extra_anchors)

    skeleton_vertices = frozenset(int(vertex) for vertex in skel_result.skeleton_vertices)
    T_confidence = frozenset(
        int(vertex)
        for vertex in skeleton_vertices
        if float(vertex_scores[int(vertex)]) >= tau_high_value
    )
    T_structural = frozenset(T_structural_raw & skeleton_vertices)
    T_global = frozenset(set(T_structural) | set(T_confidence))
    terminals_dropped_no_component = sum(1 for vertex in T_global if vertex not in component_id_of)

    bridged_mask = skel_result.skeleton_edge_mask.copy()
    steiner_added_edges_global: set[int] = set()
    component_reports: list[dict] = []
    steiner_calls = 0
    steiner_edges_added_total = 0

    parent = list(range(len(components)))

    def find(component_id: int) -> int:
        while parent[component_id] != component_id:
            parent[component_id] = parent[parent[component_id]]
            component_id = parent[component_id]
        return component_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for component_id, component_vertices in enumerate(components):
        if not T_global:
            continue
        G_reach = _bounded_search_graph(
            view=view,
            seed_vertices=component_vertices,
            r_bridge=r_bridge_value,
            probabilities=probs,
            skeleton_edge_mask=skel_result.skeleton_edge_mask,
            epsilon=epsilon_value,
        )
        reachable_terminals = T_global & set(int(vertex) for vertex in G_reach.nodes())
        for terminal in reachable_terminals:
            other_component_id = component_id_of.get(int(terminal))
            if other_component_id is not None:
                union(component_id, other_component_id)

    grouped_component_ids: dict[int, set[int]] = {}
    for component_id in range(len(components)):
        grouped_component_ids.setdefault(find(component_id), set()).add(component_id)

    for group_component_ids in sorted(grouped_component_ids.values(), key=lambda ids: min(ids)):
        comp_id = min(group_component_ids)
        comp_vertices = frozenset(
            vertex
            for component_id in sorted(group_component_ids)
            for vertex in components[component_id]
        )
        T_k = frozenset(vertex for vertex in T_global if component_id_of.get(vertex) in group_component_ids)
        T_k_structural = frozenset(vertex for vertex in T_structural if component_id_of.get(vertex) in group_component_ids)
        T_k_confidence = frozenset(vertex for vertex in T_confidence if component_id_of.get(vertex) in group_component_ids)
        skeleton_edge_count_k = int(G_skel.subgraph(comp_vertices).number_of_edges())

        report = {
            'component_id': int(comp_id),
            'skeleton_edge_count': skeleton_edge_count_k,
            'terminal_count': len(T_k),
            'terminal_count_structural': len(T_k_structural),
            'terminal_count_confidence': len(T_k_confidence),
            'sub_group_count': 0,
            'steiner_edges_added': 0,
            'skipped_reason': None,
        }

        if len(T_k) == 0:
            report['skipped_reason'] = 'no_terminals'
            component_reports.append(report)
            continue
        if len(T_k) == 1:
            report['skipped_reason'] = 'too_few_terminals'
            component_reports.append(report)
            continue

        G_search = _bounded_search_graph(
            view=view,
            seed_vertices=comp_vertices,
            r_bridge=r_bridge_value,
            probabilities=probs,
            skeleton_edge_mask=skel_result.skeleton_edge_mask,
            epsilon=epsilon_value,
        )

        assert T_k <= set(G_search.nodes()), f"component {comp_id}: terminals not in bounded search graph"

        sub_components = [frozenset(int(vertex) for vertex in sub) for sub in nx.connected_components(G_search)]
        sub_component_id_of: dict[int, int] = {}
        for sub_id, sub_component in enumerate(sub_components):
            for vertex in sub_component:
                sub_component_id_of[int(vertex)] = sub_id
        terminal_groups: dict[int, set[int]] = {}
        for terminal in T_k:
            sub_id = sub_component_id_of[int(terminal)]
            terminal_groups.setdefault(sub_id, set()).add(int(terminal))

        nontrivial_groups = [group for group in terminal_groups.values() if len(group) >= 2]
        report['sub_group_count'] = len(nontrivial_groups)

        for group in nontrivial_groups:
            any_terminal = next(iter(group))
            sub_id = sub_component_id_of[any_terminal]
            sub_vertices = sub_components[sub_id]
            G_sub = G_search.subgraph(sub_vertices).copy()

            try:
                T_steiner = nx.algorithms.approximation.steiner_tree(
                    G_sub,
                    terminal_nodes=list(group),
                    weight='weight',
                    method='mehlhorn',
                )
            except Exception as exc:
                raise RuntimeError(
                    f"steiner_tree failed for component {comp_id}, "
                    f"sub_id {sub_id}: {exc}"
                ) from exc

            steiner_calls += 1

            added_this_call = 0
            for u, v, data in T_steiner.edges(data=True):
                idx = data.get('edge_index')
                if idx is None:
                    raise RuntimeError(f"Steiner output edge ({u},{v}) missing 'edge_index'")
                edge_index = int(idx)
                if not bridged_mask[edge_index]:
                    bridged_mask[edge_index] = True
                    steiner_added_edges_global.add(edge_index)
                    added_this_call += 1

            report['steiner_edges_added'] += added_this_call
            steiner_edges_added_total += added_this_call

        component_reports.append(report)

    return BridgingResult(
        bridged_edge_mask=bridged_mask,
        steiner_added_edges=frozenset(steiner_added_edges_global),
        component_reports=tuple(component_reports),
        component_count=len(components),
        terminals_total=len(T_global),
        terminals_dropped_no_component=terminals_dropped_no_component,
        steiner_calls=steiner_calls,
        steiner_edges_added_total=steiner_edges_added_total,
        tau_high=tau_high_value,
        r_bridge=r_bridge_value,
        epsilon=epsilon_value,
    )


def diagnose_bridging_application(
    view: SeamGraphView,
    probabilities: np.ndarray,
    skel_result: SkeletonResult,
    *,
    tau_high: float = 0.70,
    r_bridge: int = 6,
    epsilon: float = 1e-3,
    anchor_boundary: bool = True,
    extra_anchor_vertices: frozenset[int] | None = None,
    topology: Any = None,
    diagnostics_threshold: float = 0.5,
) -> tuple[BridgingResult, SeamMaskDiagnostics, SeamMaskDiagnostics]:
    """
    Run Stage B and compare the before/after masks topologically.
    """
    before_probs = np.where(skel_result.skeleton_edge_mask, 1.0, 0.0).astype(np.float64, copy=False)
    before = compute_seam_mask_diagnostics(view, before_probs, threshold=diagnostics_threshold)
    bridging = compute_steiner_bridging(
        view,
        probabilities,
        skel_result,
        tau_high=tau_high,
        r_bridge=r_bridge,
        epsilon=epsilon,
        anchor_boundary=anchor_boundary,
        extra_anchor_vertices=extra_anchor_vertices,
        topology=topology,
    )
    after_probs = np.where(bridging.bridged_edge_mask, 1.0, 0.0).astype(np.float64, copy=False)
    after = compute_seam_mask_diagnostics(view, after_probs, threshold=diagnostics_threshold)
    return bridging, before, after


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
    """
    Run Stage C and compare the before/after masks topologically.
    """
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
    tau_high: float = 0.70,
    d_max: int = 3,
    r_bridge: int = 6,
    l_min: int = 4,
    epsilon: float = 1e-3,
    anchor_boundary: bool = True,
    extra_anchor_vertices: frozenset[int] | None = None,
    topology: Any = None,
) -> TopologyPipelineResult:
    """
    Run the full Stage A -> Stage B -> Stage C topology pipeline and
    return the final mask + per-stage telemetry.

    Parameters mirror the per-stage functions exactly; the same anchor
    configuration is forwarded to each stage.
    """
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
    bridge = compute_steiner_bridging(
        view,
        probabilities,
        skel,
        tau_high=tau_high,
        r_bridge=r_bridge,
        epsilon=epsilon,
        anchor_boundary=anchor_boundary,
        extra_anchor_vertices=extra_anchor_vertices,
        topology=topology,
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
        tau_high=float(tau_high),
        d_max=int(d_max),
        r_bridge=int(r_bridge),
        l_min=int(l_min),
        epsilon=float(epsilon),
        anchor_boundary=bool(anchor_boundary),
    )


def topology_pipeline_result_to_json_dict(
    result: TopologyPipelineResult,
) -> dict:
    """
    Convert a TopologyPipelineResult to a JSON-serializable dict.
    """
    skeleton = result.skeleton_result
    bridging = result.bridging_result
    pruning = result.pruning_result
    payload = {
        'bridging': {
            'component_count': int(bridging.component_count),
            'component_reports': [dict(report) for report in bridging.component_reports],
            'epsilon': float(bridging.epsilon),
            'r_bridge': int(bridging.r_bridge),
            'steiner_added_edges_count': int(len(bridging.steiner_added_edges)),
            'steiner_calls': int(bridging.steiner_calls),
            'steiner_edges_added_total': int(bridging.steiner_edges_added_total),
            'tau_high': float(bridging.tau_high),
            'terminals_dropped_no_component': int(bridging.terminals_dropped_no_component),
            'terminals_total': int(bridging.terminals_total),
        },
        'final_edge_count': int(np.count_nonzero(result.final_edge_mask)),
        'parameters': {
            'anchor_boundary': bool(result.anchor_boundary),
            'd_max': int(result.d_max),
            'epsilon': float(result.epsilon),
            'l_min': int(result.l_min),
            'r_bridge': int(result.r_bridge),
            'tau_high': float(result.tau_high),
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
