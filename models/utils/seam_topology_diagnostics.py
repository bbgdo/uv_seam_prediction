from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np

from models.utils.seam_topology_view import SeamGraphView


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
        increment_bucket(component_size_histogram, bucket_component_size(edge_count))
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
            branch_length = branch_length_from_leaf(component_graph, int(leaf))
            increment_bucket(branch_length_histogram, bucket_branch_length(branch_length))

    gap_distance_histogram: dict[str, int] = {}
    if len(component_nodes) > 1:
        ranked_components = sorted(
            zip(component_nodes, component_edge_counts, component_representatives),
            key=lambda item: (-item[1], item[2]),
        )
        selected_components = [nodes for nodes, _, _ in ranked_components[:min(20, len(ranked_components))]]
        for left_index in range(len(selected_components)):
            for right_index in range(left_index + 1, len(selected_components)):
                distance = minimum_vertex_set_distance(
                    view.vertex_graph,
                    selected_components[left_index],
                    selected_components[right_index],
                )
                if distance is None:
                    continue
                increment_bucket(gap_distance_histogram, bucket_gap_distance(distance))

    seam_probabilities = probs[seam_mask]
    outside_probabilities = probs[~seam_mask]
    return SeamMaskDiagnostics(
        threshold=threshold_value,
        seam_edge_count=seam_edge_count,
        seam_vertex_count=seam_vertex_count,
        component_count=len(component_nodes),
        component_size_histogram=ordered_bucket_dict(component_size_histogram, _COMPONENT_SIZE_BUCKETS),
        vertex_degree_histogram=dict(sorted(degree_histogram.items())),
        junction_count=junction_count,
        isolated_edge_count=isolated_edge_count,
        branch_length_histogram=ordered_bucket_dict(branch_length_histogram, _BRANCH_LENGTH_BUCKETS),
        branch_count=branch_count,
        gap_distance_histogram=ordered_bucket_dict(gap_distance_histogram, _GAP_DISTANCE_BUCKETS),
        thick_band_edge_count=thick_band_edge_count,
        mean_probability_in_seam=float(seam_probabilities.mean()) if seam_probabilities.size else 0.0,
        mean_probability_outside_seam=float(outside_probabilities.mean()) if outside_probabilities.size else 0.0,
    )


def diagnostics_to_json_dict(d: SeamMaskDiagnostics) -> dict:
    payload = {
        'branch_count': int(d.branch_count),
        'branch_length_histogram': ordered_bucket_dict(d.branch_length_histogram, _BRANCH_LENGTH_BUCKETS),
        'component_count': int(d.component_count),
        'component_size_histogram': ordered_bucket_dict(d.component_size_histogram, _COMPONENT_SIZE_BUCKETS),
        'gap_distance_histogram': ordered_bucket_dict(d.gap_distance_histogram, _GAP_DISTANCE_BUCKETS),
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


def branch_length_from_leaf(graph: nx.Graph, leaf: int) -> int:
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


def minimum_vertex_set_distance(
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


def bucket_component_size(size: int) -> str:
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


def bucket_branch_length(length: int) -> str:
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


def bucket_gap_distance(distance: int) -> str:
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


def increment_bucket(histogram: dict[str, int], bucket: str) -> None:
    histogram[bucket] = histogram.get(bucket, 0) + 1


def ordered_bucket_dict(histogram: dict[str, int], bucket_order: tuple[str, ...]) -> dict[str, int]:
    return {
        bucket: int(histogram[bucket])
        for bucket in bucket_order
        if histogram.get(bucket, 0) > 0
    }
