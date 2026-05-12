from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from preprocessing.topology import CanonicalTopology


@dataclass(frozen=True)
class SeamGraphView:
    vertex_count: int
    edge_count: int
    unique_edges: np.ndarray
    edge_lengths: np.ndarray
    canonical_vertices: np.ndarray
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
        canonical_vertices=canonical_vertices.copy(),
        vertex_to_edges=vertex_to_edges,
        vertex_graph=vertex_graph,
    )
