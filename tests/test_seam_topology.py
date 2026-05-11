import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import networkx as nx
import numpy as np

from models.utils.seam_topology import (
    BridgingResult,
    SeamGraphView,
    SkeletonResult,
    TopologyPipelineResult,
    apply_topology_pipeline,
    build_seam_graph_view,
    boundary_vertices_from_topology,
    compute_endpoint_bridging,
    compute_spur_pruning,
    compute_topology_preserving_skeleton,
    compute_seam_mask_diagnostics,
    diagnose_pruning_application,
    diagnose_skeleton_application,
    diagnostics_to_json_dict,
    lift_edge_probabilities_to_vertices,
    topology_pipeline_result_to_json_dict,
)
from preprocessing.obj_parser import ObjCorner, ObjFace, ObjMesh
from preprocessing.topology import CanonicalTopology, WeldConfig, build_topology


def _make_mesh(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> ObjMesh:
    return ObjMesh(
        vertices=tuple(vertices),
        uvs=(),
        normals=(),
        faces=tuple(
            ObjFace(
                corners=tuple(ObjCorner(vertex_index=index) for index in face),
                line_number=line_number,
            )
            for line_number, face in enumerate(faces, start=1)
        ),
        file_path=None,
    )


def _make_stub_topology(
    vertices: list[tuple[float, float, float]],
    edges: list[tuple[int, int]],
):
    canonical_edges = tuple(tuple(sorted(edge)) for edge in edges)
    return SimpleNamespace(
        canonical_vertices=tuple(vertices),
        canonical_edges=canonical_edges,
    )


def _build_view_from_topology(topology) -> tuple[object, object]:
    unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64).reshape((-1, 2))
    return topology, build_seam_graph_view(topology, unique_edges)


def _build_view_from_faces(
    faces: list[tuple[int, int, int]],
    coords: list[tuple[float, float, float]],
) -> tuple[SeamGraphView, CanonicalTopology]:
    topology = build_topology(_make_mesh(coords, faces), WeldConfig.exact())
    unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64).reshape((-1, 2))
    view = build_seam_graph_view(topology, unique_edges)
    return view, topology


def _edge_probability_vector(view, seam_probabilities: dict[tuple[int, int], float], default: float = 0.0) -> np.ndarray:
    probabilities = np.full(view.edge_count, default, dtype=np.float64)
    edge_to_index = {
        (int(edge[0]), int(edge[1])): index
        for index, edge in enumerate(view.unique_edges)
    }
    for edge, value in seam_probabilities.items():
        edge_key = tuple(sorted(edge))
        probabilities[edge_to_index[edge_key]] = float(value)
    return probabilities


def _strip_topology(length: int = 5):
    top_vertices = [(float(index), 1.0, 0.0) for index in range(length + 1)]
    bottom_vertices = [(float(index), 0.0, 0.0) for index in range(length + 1)]
    vertices = top_vertices + bottom_vertices
    offset = length + 1
    faces: list[tuple[int, int, int]] = []
    for index in range(length):
        faces.append((index, offset + index, index + 1))
        faces.append((index + 1, offset + index, offset + index + 1))
    topology = build_topology(_make_mesh(vertices, faces), WeldConfig.exact())
    return _build_view_from_topology(topology)


def _component_count(view, vertices: frozenset[int]) -> int:
    if not vertices:
        return 0
    return sum(1 for _ in nx.connected_components(view.vertex_graph.subgraph(vertices)))


def _distance_to_vertex_set(graph: nx.Graph, source: int, targets: frozenset[int]) -> int | None:
    if source in targets:
        return 0

    queue = [(source, 0)]
    visited = {source}
    while queue:
        current, distance = queue.pop(0)
        for neighbor in graph.neighbors(current):
            neighbor_index = int(neighbor)
            if neighbor_index in visited:
                continue
            if neighbor_index in targets:
                return distance + 1
            visited.add(neighbor_index)
            queue.append((neighbor_index, distance + 1))
    return None


def _edge_index(view: SeamGraphView, edge: tuple[int, int]) -> int:
    edge_key = tuple(sorted(edge))
    for index, candidate in enumerate(view.unique_edges):
        if (int(candidate[0]), int(candidate[1])) == edge_key:
            return int(index)
    raise AssertionError(f'edge {edge_key} not found')


def _grid_view(
    row_count: int,
    col_count: int,
) -> tuple[SeamGraphView, CanonicalTopology]:
    coords = [
        (float(col), float(row_count - 1 - row), 0.0)
        for row in range(row_count)
        for col in range(col_count)
    ]
    faces: list[tuple[int, int, int]] = []
    for row in range(row_count - 1):
        for col in range(col_count - 1):
            a = row * col_count + col
            b = a + 1
            c = a + col_count
            d = c + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    return _build_view_from_faces(faces, coords)


def _chain_view(vertex_count: int) -> tuple[SeamGraphView, CanonicalTopology]:
    vertices = [(float(index), 0.0, 0.0) for index in range(vertex_count)]
    faces: list[tuple[int, int, int]] = []
    for index in range(vertex_count - 1):
        aux = len(vertices)
        vertices.append((float(index) + 0.5, 1.0, 0.0))
        faces.append((index, index + 1, aux))
    return _build_view_from_faces(faces, vertices)


def _manual_skeleton_result(
    view: SeamGraphView,
    probabilities: np.ndarray,
    skeleton_edges: set[tuple[int, int]],
    *,
    skeleton_vertices: frozenset[int] | None = None,
) -> SkeletonResult:
    skeleton_edge_mask = np.zeros(view.edge_count, dtype=bool)
    touched_vertices: set[int] = set()
    for edge in skeleton_edges:
        idx = _edge_index(view, edge)
        skeleton_edge_mask[idx] = True
        touched_vertices.update(int(vertex) for vertex in edge)
    vertex_scores = lift_edge_probabilities_to_vertices(view, probabilities)
    vertices = frozenset(touched_vertices) if skeleton_vertices is None else skeleton_vertices
    return SkeletonResult(
        initial_candidate_vertices=vertices,
        anchor_vertices=frozenset(),
        skeleton_vertices=vertices,
        skeleton_edge_mask=skeleton_edge_mask,
        vertex_scores=vertex_scores,
        iterations_performed=0,
        removals_committed=0,
        refused_by_anchor=0,
        refused_by_simple_test=0,
        refused_by_distance_test=0,
        tau_low=0.30,
        d_max=3,
        anchor_boundary=False,
    )


class SeamTopologyTests(unittest.TestCase):
    def test_empty_mesh(self):
        topology = build_topology(_make_mesh([], []), WeldConfig.exact())
        _, view = _build_view_from_topology(topology)

        diagnostics = compute_seam_mask_diagnostics(view, np.zeros(0, dtype=np.float64), 0.5)

        self.assertEqual(diagnostics.seam_edge_count, 0)
        self.assertEqual(diagnostics.seam_vertex_count, 0)
        self.assertEqual(diagnostics.component_count, 0)
        self.assertEqual(diagnostics.component_size_histogram, {})
        self.assertEqual(diagnostics.vertex_degree_histogram, {})
        self.assertEqual(diagnostics.branch_length_histogram, {})
        self.assertEqual(diagnostics.gap_distance_histogram, {})
        self.assertEqual(diagnostics_to_json_dict(diagnostics)['component_size_histogram'], {})

    def test_single_isolated_edge(self):
        topology = _make_stub_topology(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            edges=[(0, 1)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.asarray([0.9], dtype=np.float64)

        diagnostics = compute_seam_mask_diagnostics(view, probabilities, 0.5)

        self.assertEqual(diagnostics.component_count, 1)
        self.assertEqual(diagnostics.isolated_edge_count, 1)
        self.assertEqual(diagnostics.branch_count, 2)
        self.assertEqual(diagnostics.branch_length_histogram, {'1': 2})
        self.assertEqual(diagnostics.component_size_histogram, {'1': 1})

    def test_clean_open_path(self):
        topology = _make_stub_topology(
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.full(view.edge_count, 0.9, dtype=np.float64)

        diagnostics = compute_seam_mask_diagnostics(view, probabilities, 0.5)

        self.assertEqual(diagnostics.component_count, 1)
        self.assertEqual(diagnostics.junction_count, 0)
        self.assertEqual(diagnostics.vertex_degree_histogram, {1: 2, 2: 3})
        self.assertEqual(diagnostics.branch_count, 2)
        self.assertEqual(diagnostics.branch_length_histogram, {'4-5': 2})
        self.assertEqual(diagnostics.component_size_histogram, {'3-5': 1})

    def test_clean_closed_loop(self):
        topology = _make_stub_topology(
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.full(view.edge_count, 0.95, dtype=np.float64)

        diagnostics = compute_seam_mask_diagnostics(view, probabilities, 0.5)

        self.assertEqual(diagnostics.component_count, 1)
        self.assertEqual(diagnostics.vertex_degree_histogram, {2: 4})
        self.assertEqual(diagnostics.junction_count, 0)
        self.assertEqual(diagnostics.branch_count, 0)
        self.assertEqual(diagnostics.branch_length_histogram, {})

    def test_thick_band_two_lane(self):
        topology, view = _strip_topology(length=5)
        offset = 6
        seam_edges = {
            (index, index + 1): 0.9
            for index in range(5)
        }
        seam_edges.update({
            (offset + index, offset + index + 1): 0.9
            for index in range(5)
        })
        seam_edges.update({
            (index, offset + index): 0.9
            for index in range(6)
        })
        probabilities = _edge_probability_vector(view, seam_edges, default=0.1)

        diagnostics = compute_seam_mask_diagnostics(view, probabilities, 0.5)

        self.assertEqual(diagnostics.component_count, 1)
        self.assertGreater(diagnostics.thick_band_edge_count, 0)
        self.assertIn(3, diagnostics.vertex_degree_histogram)
        self.assertGreater(diagnostics.vertex_degree_histogram[3], 0)

    def test_two_components_with_gap(self):
        _, view = _strip_topology(length=5)
        probabilities = _edge_probability_vector(
            view,
            {
                (0, 1): 0.9,
                (4, 5): 0.9,
            },
            default=0.1,
        )

        diagnostics = compute_seam_mask_diagnostics(view, probabilities, 0.5)

        self.assertEqual(diagnostics.component_count, 2)
        self.assertEqual(diagnostics.gap_distance_histogram, {'3': 1})

    def test_validation_errors(self):
        topology = _make_stub_topology(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            edges=[(0, 1)],
        )
        _, view = _build_view_from_topology(topology)

        with self.assertRaisesRegex(ValueError, 'finite'):
            compute_seam_mask_diagnostics(view, np.asarray([np.nan], dtype=np.float64), 0.5)
        with self.assertRaisesRegex(ValueError, r'\[0\.0, 1\.0\]'):
            compute_seam_mask_diagnostics(view, np.asarray([1.2], dtype=np.float64), 0.5)
        with self.assertRaisesRegex(ValueError, r'\[0\.0, 1\.0\]'):
            compute_seam_mask_diagnostics(view, np.asarray([0.9], dtype=np.float64), -0.1)
        with self.assertRaisesRegex(ValueError, 'shape'):
            compute_seam_mask_diagnostics(view, np.zeros(2, dtype=np.float64), 0.5)
        with self.assertRaisesRegex(ValueError, 'must match topology.canonical_edges'):
            build_seam_graph_view(
                topology,
                np.asarray([(1, 0)], dtype=np.int64),
            )


class SkeletonTests(unittest.TestCase):
    @staticmethod
    def _make_two_row_band_view() -> tuple[SeamGraphView, CanonicalTopology]:
        coords = [
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
            (3.0, 1.0, 0.0),
            (4.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
        ]
        faces = [
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 8),
            (3, 8, 4),
            (4, 8, 9),
        ]
        return _build_view_from_faces(faces, coords)

    def test_skeleton_validation_errors(self):
        topology = _make_stub_topology(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            edges=[(0, 1)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.asarray([0.4], dtype=np.float64)

        with self.assertRaises(ValueError):
            compute_topology_preserving_skeleton(view, probabilities, tau_low=-0.1)
        with self.assertRaises(ValueError):
            compute_topology_preserving_skeleton(view, probabilities, tau_low=1.1)
        with self.assertRaises(ValueError):
            compute_topology_preserving_skeleton(view, probabilities)
        with self.assertRaises(ValueError):
            compute_topology_preserving_skeleton(view, probabilities, d_max=0)
        with self.assertRaises(ValueError):
            compute_topology_preserving_skeleton(view, np.zeros(2, dtype=np.float64))
        with self.assertRaises(ValueError):
            compute_topology_preserving_skeleton(view, np.asarray([np.nan], dtype=np.float64))
        with self.assertRaises(ValueError):
            compute_topology_preserving_skeleton(
                view,
                probabilities,
                anchor_boundary=False,
                extra_anchor_vertices=frozenset({-1}),
            )
        with self.assertRaises(ValueError):
            compute_topology_preserving_skeleton(
                view,
                probabilities,
                anchor_boundary=False,
                extra_anchor_vertices=frozenset({view.vertex_count}),
            )

    def test_skeleton_lift_takes_max_over_incident_edges(self):
        topology = _make_stub_topology(
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
            ],
            edges=[(0, 1), (1, 2)],
        )
        _, view = _build_view_from_topology(topology)

        scores = lift_edge_probabilities_to_vertices(view, np.asarray([0.3, 0.8], dtype=np.float64))

        self.assertEqual(scores.dtype, np.float64)
        self.assertAlmostEqual(scores[1], 0.8)

    def test_skeleton_no_op_on_clean_isolated_chain_with_no_alternatives(self):
        middle_row = [5, 0, 3, 2, 4, 1, 6]
        top_row = [7, 8, 9, 10, 11, 12, 13]
        bottom_row = [14, 15, 16, 17, 18, 19, 20]
        coords = [
            (1.0, 1.0, 0.0),
            (5.0, 1.0, 0.0),
            (3.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
            (4.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (6.0, 1.0, 0.0),
            (0.0, 2.0, 0.0),
            (1.0, 2.0, 0.0),
            (2.0, 2.0, 0.0),
            (3.0, 2.0, 0.0),
            (4.0, 2.0, 0.0),
            (5.0, 2.0, 0.0),
            (6.0, 2.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (6.0, 0.0, 0.0),
        ]
        faces: list[tuple[int, int, int]] = []
        for col in range(6):
            faces.append((top_row[col], middle_row[col], top_row[col + 1]))
            faces.append((top_row[col + 1], middle_row[col], middle_row[col + 1]))
            faces.append((middle_row[col], bottom_row[col], middle_row[col + 1]))
            faces.append((middle_row[col + 1], bottom_row[col], bottom_row[col + 1]))
        view, _ = _build_view_from_faces(faces, coords)
        probabilities = _edge_probability_vector(
            view,
            {
                (0, 3): 0.95,
                (2, 3): 0.95,
                (2, 4): 0.95,
                (1, 4): 0.95,
            },
            default=0.0,
        )

        result = compute_topology_preserving_skeleton(
            view,
            probabilities,
            tau_low=0.3,
            d_max=3,
            anchor_boundary=False,
        )

        self.assertEqual(result.anchor_vertices, frozenset())
        self.assertGreater(result.removals_committed, 0)
        self.assertNotIn(0, result.skeleton_vertices)
        self.assertNotIn(1, result.skeleton_vertices)
        self.assertIn(2, result.skeleton_vertices)
        self.assertLess(len(result.skeleton_vertices), len(result.initial_candidate_vertices))

    def test_skeleton_thins_thick_band_to_one_dimensional(self):
        view, topology = self._make_two_row_band_view()
        low_probabilities = np.full(view.edge_count, 0.4, dtype=np.float64)
        saturated_probabilities = np.full(view.edge_count, 0.99, dtype=np.float64)

        low_result, low_before, low_after = diagnose_skeleton_application(
            view,
            low_probabilities,
            tau_low=0.3,
            d_max=3,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({0, 4}),
            topology=topology,
        )
        saturated_result, saturated_before, saturated_after = diagnose_skeleton_application(
            view,
            saturated_probabilities,
            tau_low=0.3,
            d_max=3,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({0, 4}),
            topology=topology,
        )

        self.assertGreater(low_result.refused_by_simple_test, 0)
        self.assertGreater(low_result.removals_committed, 0)
        self.assertGreater(low_before.thick_band_edge_count, 0)
        self.assertEqual(low_after.thick_band_edge_count, 0)
        self.assertEqual(saturated_before.thick_band_edge_count, low_before.thick_band_edge_count)
        self.assertEqual(saturated_after.thick_band_edge_count, 0)
        self.assertEqual(low_result.skeleton_vertices, saturated_result.skeleton_vertices)
        self.assertTrue(np.array_equal(low_result.skeleton_edge_mask, saturated_result.skeleton_edge_mask))
        self.assertIn(0, low_result.skeleton_vertices)
        self.assertIn(4, low_result.skeleton_vertices)
        skeleton_graph = view.vertex_graph.subgraph(low_result.skeleton_vertices)
        self.assertTrue(nx.has_path(skeleton_graph, 0, 4))

    def test_skeleton_preserves_anchor_only_chain(self):
        topology = _make_stub_topology(
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (5.0, 0.0, 0.0),
                (6.0, 0.0, 0.0),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.full(view.edge_count, 0.4, dtype=np.float64)

        result = compute_topology_preserving_skeleton(
            view,
            probabilities,
            tau_low=0.3,
            d_max=3,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({0, 6}),
        )

        self.assertEqual(result.initial_candidate_vertices, frozenset(range(7)))
        self.assertEqual(result.anchor_vertices, frozenset({0, 6}))
        self.assertEqual(result.removals_committed, 0)
        self.assertEqual(result.skeleton_vertices, frozenset(range(7)))
        self.assertEqual(result.refused_by_simple_test, 5)

    def test_skeleton_distance_constraint_blocks_overthinning(self):
        coords: list[tuple[float, float, float]] = []
        row_count = 4
        col_count = 9
        for row in range(row_count):
            y = float(row_count - 1 - row)
            for col in range(col_count):
                coords.append((float(col), y, 0.0))
        faces: list[tuple[int, int, int]] = []
        for row in range(row_count - 1):
            for col in range(col_count - 1):
                top_left = (row * col_count) + col
                bottom_left = ((row + 1) * col_count) + col
                if (row + col) % 2 == 0:
                    faces.append((top_left, bottom_left, top_left + 1))
                    faces.append((top_left + 1, bottom_left, bottom_left + 1))
                else:
                    faces.append((top_left, bottom_left, bottom_left + 1))
                    faces.append((top_left, bottom_left + 1, top_left + 1))
        view, topology = _build_view_from_faces(faces, coords)
        probabilities = np.full(view.edge_count, 0.4, dtype=np.float64)

        result, _, after = diagnose_skeleton_application(
            view,
            probabilities,
            tau_low=0.3,
            d_max=2,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({0, col_count - 1, (row_count - 1) * col_count, (row_count * col_count) - 1}),
            topology=topology,
        )

        self.assertGreater(result.refused_by_distance_test, 0)
        self.assertEqual(after.thick_band_edge_count, 0)
        for vertex in sorted(result.initial_candidate_vertices - result.skeleton_vertices):
            distance = _distance_to_vertex_set(view.vertex_graph, vertex, result.skeleton_vertices)
            self.assertIsNotNone(distance)
            self.assertLessEqual(distance, 2)

    def test_skeleton_anchor_protection_invariant(self):
        rng = np.random.default_rng(0)
        top_row = [(float(index), 1.0, 0.0) for index in range(6)]
        middle_row = [(float(index), 0.5, 0.0) for index in range(6)]
        bottom_row = [(float(index), 0.0, 0.0) for index in range(6)]
        coords = top_row + middle_row + bottom_row
        faces: list[tuple[int, int, int]] = []
        row_width = 6
        for row in range(2):
            row_offset = row * row_width
            next_offset = (row + 1) * row_width
            for col in range(row_width - 1):
                faces.append((row_offset + col, next_offset + col, row_offset + col + 1))
                faces.append((row_offset + col + 1, next_offset + col, next_offset + col + 1))
        view, topology = _build_view_from_faces(faces, coords)
        probabilities = rng.random(view.edge_count, dtype=np.float64)
        extra_anchors = frozenset(int(value) for value in rng.choice(view.vertex_count, size=4, replace=False))

        result = compute_topology_preserving_skeleton(
            view,
            probabilities,
            tau_low=0.3,
            d_max=3,
            anchor_boundary=True,
            extra_anchor_vertices=extra_anchors,
            topology=topology,
        )

        vertex_scores = lift_edge_probabilities_to_vertices(view, probabilities)
        expected_candidates = frozenset(int(index) for index in np.flatnonzero(vertex_scores >= 0.3))
        expected_anchors = frozenset(
            set(boundary_vertices_from_topology(topology))
            | {vertex for vertex in extra_anchors if vertex in expected_candidates}
        ) & expected_candidates
        self.assertEqual(result.anchor_vertices, expected_anchors)
        self.assertTrue(result.anchor_vertices <= result.skeleton_vertices)

    def test_skeleton_components_invariant(self):
        rng = np.random.default_rng(1)
        top_row = [(float(index), 1.0, 0.0) for index in range(6)]
        middle_row = [(float(index), 0.5, 0.0) for index in range(6)]
        bottom_row = [(float(index), 0.0, 0.0) for index in range(6)]
        coords = top_row + middle_row + bottom_row
        faces: list[tuple[int, int, int]] = []
        row_width = 6
        for row in range(2):
            row_offset = row * row_width
            next_offset = (row + 1) * row_width
            for col in range(row_width - 1):
                faces.append((row_offset + col, next_offset + col, row_offset + col + 1))
                faces.append((row_offset + col + 1, next_offset + col, next_offset + col + 1))
        view, topology = _build_view_from_faces(faces, coords)
        probabilities = rng.random(view.edge_count, dtype=np.float64)
        extra_anchors = frozenset(int(value) for value in rng.choice(view.vertex_count, size=3, replace=False))

        result = compute_topology_preserving_skeleton(
            view,
            probabilities,
            tau_low=0.3,
            d_max=3,
            anchor_boundary=True,
            extra_anchor_vertices=extra_anchors,
            topology=topology,
        )

        self.assertEqual(
            _component_count(view, result.initial_candidate_vertices),
            _component_count(view, result.skeleton_vertices),
        )

    def test_skeleton_isolated_vertex_below_threshold(self):
        topology = _make_stub_topology(
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
            ],
            edges=[(0, 1), (1, 2)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.asarray([0.1, 0.8], dtype=np.float64)

        result = compute_topology_preserving_skeleton(
            view,
            probabilities,
            tau_low=0.3,
            d_max=3,
            anchor_boundary=False,
        )

        self.assertNotIn(0, result.initial_candidate_vertices)
        self.assertNotIn(0, result.skeleton_vertices)

    def test_diagnose_skeleton_application_reduces_thick_bands(self):
        view, topology = self._make_two_row_band_view()
        probabilities = np.full(view.edge_count, 0.4, dtype=np.float64)
        saturated_probabilities = np.full(view.edge_count, 0.99, dtype=np.float64)

        result, before, after = diagnose_skeleton_application(
            view,
            probabilities,
            tau_low=0.3,
            d_max=3,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({0, 4}),
            topology=topology,
        )
        saturated_result, _, saturated_after = diagnose_skeleton_application(
            view,
            saturated_probabilities,
            tau_low=0.3,
            d_max=3,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({0, 4}),
            topology=topology,
        )

        self.assertGreater(before.thick_band_edge_count, 0)
        self.assertEqual(after.thick_band_edge_count, 0)
        self.assertGreater(result.removals_committed, 0)
        self.assertEqual(saturated_after.thick_band_edge_count, 0)
        self.assertGreater(saturated_result.removals_committed, 0)

    def test_skeleton_thins_saturated_thick_band(self):
        coords = [
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
            (3.0, 1.0, 0.0),
            (4.0, 1.0, 0.0),
            (5.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
        ]
        faces = [
            (0, 6, 1),
            (1, 6, 7),
            (1, 7, 2),
            (2, 7, 8),
            (2, 8, 3),
            (3, 8, 9),
            (3, 9, 4),
            (4, 9, 10),
            (4, 10, 5),
            (5, 10, 11),
        ]
        view, topology = _build_view_from_faces(faces, coords)
        probabilities = np.full(view.edge_count, 0.99, dtype=np.float64)

        result = compute_topology_preserving_skeleton(
            view,
            probabilities,
            tau_low=0.3,
            d_max=3,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({0, 5}),
            topology=topology,
        )
        diagnostics_after = compute_seam_mask_diagnostics(
            view,
            np.where(result.skeleton_edge_mask, 1.0, 0.0).astype(np.float64, copy=False),
            threshold=0.5,
        )

        self.assertGreater(result.removals_committed, 0)
        self.assertEqual(diagnostics_after.thick_band_edge_count, 0)
        self.assertIn(0, result.skeleton_vertices)
        self.assertIn(5, result.skeleton_vertices)
        skeleton_graph = view.vertex_graph.subgraph(result.skeleton_vertices)
        self.assertTrue(nx.has_path(skeleton_graph, 0, 5))

    def test_boundary_vertices_from_topology(self):
        coords = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        faces = [
            (0, 1, 2),
            (0, 2, 3),
        ]
        _, topology = _build_view_from_faces(faces, coords)

        self.assertEqual(boundary_vertices_from_topology(topology), frozenset({0, 1, 2, 3}))
        self.assertEqual(boundary_vertices_from_topology(None), frozenset())


def _manual_pipeline_result(
    *,
    skel_result: SkeletonResult,
    bridging_result: BridgingResult,
    pruning_result,
    final_edge_mask: np.ndarray,
) -> TopologyPipelineResult:
    return TopologyPipelineResult(
        final_edge_mask=final_edge_mask,
        skeleton_result=skel_result,
        bridging_result=bridging_result,
        pruning_result=pruning_result,
        tau_low=0.30,
        d_max=3,
        r_bridge=6,
        l_min=4,
        anchor_boundary=False,
    )


class EndpointBridgingTests(unittest.TestCase):
    @staticmethod
    def _chain_view(vertex_count: int) -> tuple[SeamGraphView, CanonicalTopology]:
        return _chain_view(vertex_count)

    @staticmethod
    def _chain_skeleton(
        vertex_count: int,
        skeleton_edges: set[tuple[int, int]],
    ) -> tuple[SeamGraphView, np.ndarray, SkeletonResult]:
        view, _ = EndpointBridgingTests._chain_view(vertex_count)
        probabilities = _edge_probability_vector(
            view,
            {edge: 0.95 for edge in skeleton_edges},
            default=0.01,
        )
        return view, probabilities, _manual_skeleton_result(view, probabilities, skeleton_edges)

    @staticmethod
    def _mask_edges(view: SeamGraphView, mask: np.ndarray) -> set[tuple[int, int]]:
        return {
            (int(view.unique_edges[index, 0]), int(view.unique_edges[index, 1]))
            for index in np.flatnonzero(mask)
        }

    def test_endpoint_bridging_closes_one_edge_gap(self):
        skeleton_edges = {(0, 1), (1, 2), (3, 4), (4, 5)}
        view, _, skel_result = self._chain_skeleton(6, skeleton_edges)

        result = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=1,
            max_bridge_euclidean_ratio=1.0,
        )

        gap_index = _edge_index(view, (2, 3))
        self.assertTrue(result.bridged_edge_mask[gap_index])
        self.assertEqual(result.added_bridge_edges, frozenset({gap_index}))
        self.assertEqual(result.bridges_accepted, 1)
        self.assertEqual(result.added_bridge_edges_count, 1)
        self.assertEqual(result.endpoints_before, 4)
        self.assertEqual(result.components_after, 1)
        self.assertEqual(result.accepted_bridge_reports[0]['bridge_edge_count'], 1)

    def test_endpoint_bridging_closes_two_edge_gap(self):
        skeleton_edges = {(0, 1), (1, 2), (4, 5), (5, 6)}
        view, _, skel_result = self._chain_skeleton(7, skeleton_edges)

        result = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=2,
            max_bridge_euclidean_ratio=1.0,
        )

        expected = {_edge_index(view, (2, 3)), _edge_index(view, (3, 4))}
        self.assertTrue(all(result.bridged_edge_mask[index] for index in expected))
        self.assertEqual(result.added_bridge_edges, frozenset(expected))
        self.assertEqual(result.max_bridge_length_edges, 2)
        self.assertEqual(result.mean_bridge_length_edges, 2.0)
        self.assertEqual(result.bridge_length_edges_histogram, {2: 1})

    def test_endpoint_bridging_rejects_distant_endpoints(self):
        skeleton_edges = {(0, 1), (1, 2), (9, 10), (10, 11)}
        view, _, skel_result = self._chain_skeleton(12, skeleton_edges)

        by_graph = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=3,
            max_bridge_euclidean_ratio=1.0,
        )
        by_euclidean = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=8,
            max_bridge_euclidean_ratio=0.01,
        )

        self.assertEqual(by_graph.bridges_accepted, 0)
        self.assertGreater(by_graph.bridges_rejected_by_graph_length, 0)
        self.assertEqual(by_euclidean.bridges_accepted, 0)
        self.assertGreater(by_euclidean.bridges_rejected_by_euclidean_distance, 0)

    def test_endpoint_bridging_cross_case_chooses_local_mutual_pairs(self):
        skeleton_edges = {(0, 1), (3, 4), (10, 11), (13, 14)}
        view, _, skel_result = self._chain_skeleton(15, skeleton_edges)

        result = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=2,
            max_bridge_euclidean_ratio=1.0,
        )

        expected_added = {
            _edge_index(view, (1, 2)),
            _edge_index(view, (2, 3)),
            _edge_index(view, (11, 12)),
            _edge_index(view, (12, 13)),
        }
        self.assertEqual(result.added_bridge_edges, frozenset(expected_added))
        self.assertEqual(result.bridges_accepted, 2)
        accepted_pairs = {
            tuple(report['endpoint_vertex_ids'])
            for report in result.accepted_bridge_reports
        }
        self.assertEqual(accepted_pairs, {(1, 3), (11, 13)})

    def test_endpoint_bridging_closed_loop_no_ops(self):
        topology = _make_stub_topology(
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (0, 3)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.full(view.edge_count, 0.95, dtype=np.float64)
        skeleton_edges = {(0, 1), (1, 2), (2, 3), (0, 3)}
        skel_result = _manual_skeleton_result(view, probabilities, skeleton_edges)

        result = compute_endpoint_bridging(view, skel_result)

        self.assertTrue(np.array_equal(result.bridged_edge_mask, skel_result.skeleton_edge_mask))
        self.assertEqual(result.endpoints_before, 0)
        self.assertEqual(result.bridges_accepted, 0)

    def test_endpoint_bridging_same_component_loop_size_rules(self):
        large_edges = {(index, index + 1) for index in range(8)}
        large_topology = _make_stub_topology(
            vertices=[(float(index), 0.0, 0.0) for index in range(9)],
            edges=sorted(large_edges | {(0, 8)}),
        )
        _, large_view = _build_view_from_topology(large_topology)
        probabilities = np.full(large_view.edge_count, 0.95, dtype=np.float64)
        large_skel = _manual_skeleton_result(large_view, probabilities, large_edges)

        large = compute_endpoint_bridging(
            large_view,
            large_skel,
            max_bridge_edges=1,
            max_bridge_euclidean_ratio=1.0,
            min_loop_size_to_allow=8,
        )

        tiny_topology = _make_stub_topology(
            vertices=[(float(index), 0.0, 0.0) for index in range(4)],
            edges=[(0, 1), (1, 2), (2, 3), (0, 3)],
        )
        _, tiny_view = _build_view_from_topology(tiny_topology)
        tiny_probs = np.full(tiny_view.edge_count, 0.95, dtype=np.float64)
        tiny_skel = _manual_skeleton_result(tiny_view, tiny_probs, {(0, 1), (1, 2), (2, 3)})
        tiny = compute_endpoint_bridging(
            tiny_view,
            tiny_skel,
            max_bridge_edges=1,
            max_bridge_euclidean_ratio=1.0,
            min_loop_size_to_allow=8,
        )

        self.assertTrue(large.bridged_edge_mask[_edge_index(large_view, (0, 8))])
        self.assertEqual(large.bridges_accepted, 1)
        self.assertEqual(tiny.bridges_accepted, 0)
        self.assertGreater(tiny.bridges_rejected_by_already_connected, 0)

    def test_endpoint_bridging_tiny_spur_is_stage_c_responsibility(self):
        skeleton_edges = {(0, 1), (1, 2), (2, 3), (1, 4)}
        topology = _make_stub_topology(
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
            ],
            edges=sorted(skeleton_edges),
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.full(view.edge_count, 0.95, dtype=np.float64)
        skel_result = _manual_skeleton_result(view, probabilities, skeleton_edges)

        result = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=2,
            max_bridge_euclidean_ratio=1.0,
        )

        self.assertEqual(result.bridges_accepted, 0)
        self.assertTrue(np.array_equal(result.bridged_edge_mask, skel_result.skeleton_edge_mask))

    def test_endpoint_bridging_rejects_path_reusing_skeleton_edges(self):
        skeleton_edges = {(0, 1), (1, 2), (2, 3), (3, 4)}
        view, _, skel_result = self._chain_skeleton(5, skeleton_edges)

        result = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=4,
            max_bridge_euclidean_ratio=1.0,
            min_loop_size_to_allow=4,
        )

        self.assertEqual(result.bridges_accepted, 0)
        self.assertGreater(result.bridges_rejected_by_existing_seam_edge, 0)

    def test_endpoint_bridging_rejects_path_through_skeleton_vertex(self):
        topology = _make_stub_topology(
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (5.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
                (2.0, -1.0, 0.0),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7), (2, 7)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.full(view.edge_count, 0.95, dtype=np.float64)
        skeleton_edges = {(0, 1), (4, 5), (2, 6), (6, 7), (2, 7)}
        skel_result = _manual_skeleton_result(view, probabilities, skeleton_edges)

        result = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=3,
            max_bridge_euclidean_ratio=1.0,
        )

        self.assertEqual(result.bridges_accepted, 0)
        self.assertGreater(result.bridges_rejected_by_skeleton_intersection, 0)

    def test_bridge_telemetry_reports_final_survival(self):
        skeleton_edges = {(0, 1), (1, 2), (3, 4), (4, 5)}
        view, probabilities, skel_result = self._chain_skeleton(6, skeleton_edges)
        bridge = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=1,
            max_bridge_euclidean_ratio=1.0,
        )
        prune = compute_spur_pruning(view, bridge, l_min=1, anchor_boundary=False)
        pipeline = _manual_pipeline_result(
            skel_result=skel_result,
            bridging_result=bridge,
            pruning_result=prune,
            final_edge_mask=prune.pruned_edge_mask,
        )

        payload = topology_pipeline_result_to_json_dict(pipeline)
        bridging = payload['bridging']

        gap_index = _edge_index(view, (2, 3))
        self.assertEqual(bridging['seam_edge_count_before_stage_b'], int(skel_result.skeleton_edge_mask.sum()))
        self.assertEqual(bridging['seam_edge_count_after_stage_b'], int(bridge.bridged_edge_mask.sum()))
        self.assertEqual(bridging['seam_edge_count_after_stage_c'], int(prune.pruned_edge_mask.sum()))
        self.assertEqual(bridging['accepted_bridge_edge_indices'], [gap_index])
        self.assertEqual(bridging['accepted_bridge_edge_keys'], [[2, 3]])
        self.assertEqual(bridging['accepted_bridge_edges_survived_to_final'], 1)
        self.assertEqual(bridging['accepted_bridge_edges_removed_by_stage_c'], 0)
        self.assertTrue(bridging['final_output_contains_accepted_bridge_edges'])
        self.assertEqual(
            bridging['bridge_edge_ids_final_presence'],
            [{
                'edge_id': gap_index,
                'vertex_ids_0based': [2, 3],
                'in_after_stage_b': True,
                'in_after_stage_c': True,
                'in_output_seam_edge_indices': True,
                'in_output_seam_edges': True,
                'original_blender_edge_if_traceable': None,
                'applied_by_blender_if_traceable': None,
            }],
        )

    def test_bridge_telemetry_reports_stage_c_removal(self):
        skeleton_edges = {(0, 1), (2, 3)}
        view, _, skel_result = self._chain_skeleton(4, skeleton_edges)
        bridge = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=1,
            max_bridge_euclidean_ratio=1.0,
        )
        prune = compute_spur_pruning(view, bridge, l_min=4, anchor_boundary=False)
        pipeline = _manual_pipeline_result(
            skel_result=skel_result,
            bridging_result=bridge,
            pruning_result=prune,
            final_edge_mask=prune.pruned_edge_mask,
        )

        bridging = topology_pipeline_result_to_json_dict(pipeline)['bridging']

        self.assertEqual(bridging['accepted_bridge_edges_survived_to_final'], 0)
        self.assertEqual(bridging['accepted_bridge_edges_removed_by_stage_c'], 1)
        self.assertFalse(bridging['final_output_contains_accepted_bridge_edges'])
        self.assertFalse(bridging['bridge_edge_ids_final_presence'][0]['in_after_stage_c'])

    def test_build_output_payload_uses_final_mask_for_seam_lists(self):
        from tools.predict_seams import build_output_payload

        topology = SimpleNamespace(
            canonical_vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            canonical_faces=(),
            edge_incidence={(0, 1): (object(), object()), (1, 2): (object(),)},
        )
        unique_edges = np.asarray([(0, 1), (1, 2)], dtype=np.int64)
        diagnostics = {
            'postprocess': {
                'bridging': {
                    'accepted_bridge_edge_indices': [1],
                    'bridge_edge_ids_final_presence': [{
                        'edge_id': 1,
                        'vertex_ids_0based': [1, 2],
                        'in_after_stage_b': True,
                        'in_after_stage_c': True,
                        'in_output_seam_edge_indices': False,
                        'in_output_seam_edges': False,
                        'original_blender_edge_if_traceable': None,
                        'applied_by_blender_if_traceable': None,
                    }],
                },
            },
        }

        payload = build_output_payload(
            mesh_path=Path('mesh.obj'),
            output_json=Path('prediction.json'),
            weights_path=Path('weights.pt'),
            config_path=Path('config.json'),
            summary_path=Path('summary.json'),
            model_type='graphsage',
            feature_bundle='paper14',
            selection=SimpleNamespace(feature_group='paper14', feature_names=(), feature_count=0),
            threshold=0.5,
            device='cpu',
            topology=topology,
            unique_edges=unique_edges,
            probabilities=np.asarray([0.9, 0.1], dtype=np.float64),
            seam_mask=np.asarray([False, True], dtype=bool),
            write_all_edges=True,
            diagnostics=diagnostics,
        )

        self.assertEqual(payload['seam_edge_indices'], [1])
        self.assertEqual([row['canonical_edge_index'] for row in payload['seam_edges']], [1])
        bridging = payload['diagnostics']['postprocess']['bridging']
        self.assertEqual(bridging['accepted_bridge_edges_survived_to_final'], 1)
        self.assertEqual(bridging['accepted_bridge_edges_removed_by_stage_c'], 0)
        self.assertTrue(bridging['bridge_edge_ids_final_presence'][0]['in_output_seam_edge_indices'])
        self.assertTrue(bridging['bridge_edge_ids_final_presence'][0]['in_output_seam_edges'])

    def test_same_component_candidate_telemetry_is_populated(self):
        topology = _make_stub_topology(
            vertices=[(float(index), 0.0, 0.0) for index in range(4)],
            edges=[(0, 1), (1, 2), (2, 3), (0, 3)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.full(view.edge_count, 0.95, dtype=np.float64)
        skel_result = _manual_skeleton_result(view, probabilities, {(0, 1), (1, 2), (2, 3)})

        result = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=1,
            max_bridge_euclidean_ratio=1.0,
            min_loop_size_to_allow=8,
        )

        self.assertEqual(result.same_component_candidates_considered, 1)
        self.assertEqual(result.same_component_bridges_accepted, 0)
        self.assertEqual(result.same_component_bridges_rejected_by_already_connected, 1)
        self.assertEqual(len(result.same_component_rejected_candidate_reports), 1)
        report = result.same_component_rejected_candidate_reports[0]
        self.assertEqual(report['endpoint_vertex_ids'], [0, 3])
        self.assertEqual(report['rejection_reason'], 'already_connected')
        self.assertEqual(report['path_edge_count'], 1)

    def test_missing_edge_continuity_diagnostic_detects_skeleton_vertices(self):
        topology = _make_stub_topology(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
            edges=[(0, 1), (1, 2), (0, 2)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.full(view.edge_count, 0.95, dtype=np.float64)
        skel_result = _manual_skeleton_result(view, probabilities, {(0, 1), (1, 2)})

        result = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=1,
            max_bridge_euclidean_ratio=1.0,
        )

        candidates = result.local_missing_edge_continuity_candidates
        self.assertEqual(result.local_missing_edge_continuity_candidates_total, 1)
        self.assertEqual(candidates[0]['vertex_ids_0based'], [0, 2])
        self.assertEqual(candidates[0]['why_not_currently_accepted'], 'both_vertices_already_in_skeleton')

    def test_endpoint_to_skeleton_diagnostic_detects_degree_two_target(self):
        topology = _make_stub_topology(
            vertices=[(float(index), 0.0, 0.0) for index in range(6)],
            edges=[(0, 1), (1, 2), (4, 5), (1, 3), (3, 5)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.full(view.edge_count, 0.95, dtype=np.float64)
        skel_result = _manual_skeleton_result(view, probabilities, {(0, 1), (1, 2), (4, 5)})

        result = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=2,
            max_bridge_euclidean_ratio=1.0,
        )

        candidates = [
            report for report in result.endpoint_to_skeleton_candidates
            if report['endpoint_vertex_id'] == 5 and report['target_skeleton_vertex_id'] == 1
        ]
        self.assertTrue(candidates)
        self.assertEqual(candidates[0]['target_seam_degree'], 2)
        self.assertEqual(candidates[0]['reason_not_covered'], 'target_is_not_degree_1_endpoint')

    def test_near_junction_diagnostic_detects_local_gap(self):
        topology = _make_stub_topology(
            vertices=[(float(index), 0.0, 0.0) for index in range(8)],
            edges=[(0, 1), (1, 2), (1, 3), (5, 6), (1, 4), (4, 5)],
        )
        _, view = _build_view_from_topology(topology)
        probabilities = np.full(view.edge_count, 0.95, dtype=np.float64)
        skel_result = _manual_skeleton_result(view, probabilities, {(0, 1), (1, 2), (1, 3), (5, 6)})

        result = compute_endpoint_bridging(
            view,
            skel_result,
            max_bridge_edges=2,
            max_bridge_euclidean_ratio=1.0,
        )

        candidates = [
            report for report in result.near_junction_gap_candidates
            if report['source_vertex_id'] == 5 and report['junction_vertex_id'] == 1
        ]
        self.assertTrue(candidates)
        self.assertEqual(candidates[0]['path_edge_count'], 2)
        self.assertEqual(result.near_junction_gap_candidates_total, len(result.near_junction_gap_candidates))


class BlenderSeamMappingDebugTests(unittest.TestCase):
    class _FakeEdge:
        def __init__(self, vertices):
            self.vertices = vertices
            self.use_seam = False

    class _FakeMesh:
        def __init__(self, edges):
            self.edges = [BlenderSeamMappingDebugTests._FakeEdge(edge) for edge in edges]
            self.update_count = 0

        def update(self):
            self.update_count += 1

    @staticmethod
    def _load_seam_mapping():
        path = Path(__file__).resolve().parents[1] / 'blender_addon' / 'uv_seam_predictor' / 'seam_mapping.py'
        spec = importlib.util.spec_from_file_location('uvsp_seam_mapping_debug', path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_seam_mapping_consumes_final_seam_edges(self):
        seam_mapping = self._load_seam_mapping()
        payload = {
            'status': 'ok',
            'seam_edge_indices': [99],
            'seam_edges': [{'canonical_edge_index': 1, 'vertex_ids_0based': [1, 2]}],
        }
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as handle:
            json.dump(payload, handle)
            json_path = handle.name
        try:
            keys = seam_mapping.load_predicted_edge_keys(json_path)
        finally:
            Path(json_path).unlink(missing_ok=True)

        self.assertEqual(keys, [(1, 2)])

    def test_seam_mapping_reports_ignored_bridge_edges(self):
        seam_mapping = self._load_seam_mapping()
        mesh = self._FakeMesh(edges=[(0, 1), (1, 2)])

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1), (2, 3)],
            clear_existing=True,
            accepted_bridge_entries=[
                {
                    'canonical_edge_index': 10,
                    'vertex_ids_0based': [0, 1],
                    'bridge_path_id': 0,
                    'path_edge_count': 1,
                    'same_component': False,
                    'present_in_final_json': True,
                },
                {
                    'canonical_edge_index': 11,
                    'vertex_ids_0based': [2, 3],
                    'bridge_path_id': 1,
                    'path_edge_count': 1,
                    'same_component': False,
                    'present_in_final_json': True,
                },
            ],
        )

        self.assertEqual(result.applied, 1)
        self.assertEqual(result.ignored_non_original, 1)
        self.assertEqual(result.accepted_bridge_edges_present_in_json, 2)
        self.assertEqual(result.accepted_bridge_edges_applied, 1)
        self.assertEqual(result.accepted_bridge_edges_ignored_non_original, 1)
        self.assertEqual(
            list(result.accepted_bridge_apply_trace),
            [
                {
                    'canonical_edge_index': 10,
                    'vertex_ids_0based': [0, 1],
                    'bridge_path_id': 0,
                    'path_edge_count': 1,
                    'same_component': False,
                    'present_in_final_json': True,
                    'blender_edge_key_exists': True,
                    'applied_to_blender': True,
                    'ignored_reason': None,
                    'duplicate_or_already_marked': False,
                },
                {
                    'canonical_edge_index': 11,
                    'vertex_ids_0based': [2, 3],
                    'bridge_path_id': 1,
                    'path_edge_count': 1,
                    'same_component': False,
                    'present_in_final_json': True,
                    'blender_edge_key_exists': False,
                    'applied_to_blender': False,
                    'ignored_reason': 'non_original',
                    'duplicate_or_already_marked': False,
                },
            ],
        )

    def test_load_accepted_bridge_edge_keys_from_prediction_json(self):
        seam_mapping = self._load_seam_mapping()
        payload = {
            'status': 'ok',
            'seam_edges': [],
            'diagnostics': {
                'postprocess': {
                    'bridging': {
                        'accepted_bridge_edge_indices': [7, 8],
                        'accepted_bridge_edge_keys': [[3, 2], [5, 6]],
                        'accepted_bridge_reports': [{
                            'path_edge_ids': [7, 8],
                            'path_edge_count': 2,
                            'same_component': True,
                        }],
                        'bridge_edge_ids_final_presence': [
                            {'edge_id': 7, 'in_output_seam_edges': True},
                            {'edge_id': 8, 'in_output_seam_edges': False},
                        ],
                    },
                },
            },
        }
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as handle:
            json.dump(payload, handle)
            json_path = handle.name
        try:
            keys = seam_mapping.load_accepted_bridge_edge_keys(json_path)
        finally:
            Path(json_path).unlink(missing_ok=True)

        self.assertEqual(keys, [(2, 3), (5, 6)])

        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as handle:
            json.dump(payload, handle)
            json_path = handle.name
        try:
            entries = seam_mapping.load_accepted_bridge_debug_entries(json_path)
        finally:
            Path(json_path).unlink(missing_ok=True)
        self.assertEqual(entries[0]['canonical_edge_index'], 7)
        self.assertEqual(entries[0]['bridge_path_id'], 0)
        self.assertEqual(entries[0]['path_edge_count'], 2)
        self.assertTrue(entries[0]['same_component'])
        self.assertTrue(entries[0]['present_in_final_json'])


class PruningTests(unittest.TestCase):
    @staticmethod
    def _view_for_seam_edges(
        seam_edges: set[tuple[int, int]],
    ) -> tuple[SeamGraphView, CanonicalTopology]:
        max_vertex = max((max(edge) for edge in seam_edges), default=-1)
        vertices = [(float(index), 0.0, 0.0) for index in range(max_vertex + 1)]
        faces: list[tuple[int, int, int]] = []
        for edge_index, edge in enumerate(sorted(tuple(sorted(edge)) for edge in seam_edges)):
            u, v = edge
            aux = len(vertices)
            ux, _, _ = vertices[u]
            vx, _, _ = vertices[v]
            vertices.append(((ux + vx) * 0.5, float(edge_index + 1), 0.0))
            faces.append((u, v, aux))
        return _build_view_from_faces(faces, vertices)

    @staticmethod
    def _grid_view(size: int = 4) -> tuple[SeamGraphView, CanonicalTopology]:
        vertices = [
            (float(x), float(y), 0.0)
            for y in range(size)
            for x in range(size)
        ]
        faces: list[tuple[int, int, int]] = []
        for y in range(size - 1):
            for x in range(size - 1):
                a = y * size + x
                b = a + 1
                c = a + size
                d = c + 1
                faces.append((a, b, c))
                faces.append((b, d, c))
        return _build_view_from_faces(faces, vertices)

    @staticmethod
    def _bridging_result(view: SeamGraphView, seam_edges: set[tuple[int, int]]) -> BridgingResult:
        mask = np.zeros(view.edge_count, dtype=bool)
        for edge in seam_edges:
            mask[_edge_index(view, edge)] = True
        return BridgingResult(
            bridged_edge_mask=mask,
            component_reports=tuple(),
            r_bridge=6,
        )

    @staticmethod
    def _mask_edges(view: SeamGraphView, mask: np.ndarray) -> set[tuple[int, int]]:
        return {
            (int(view.unique_edges[index, 0]), int(view.unique_edges[index, 1]))
            for index in np.flatnonzero(mask)
        }

    @staticmethod
    def _masked_graph(view: SeamGraphView, mask: np.ndarray) -> nx.Graph:
        graph = nx.Graph()
        for index in np.flatnonzero(mask):
            u = int(view.unique_edges[index, 0])
            v = int(view.unique_edges[index, 1])
            graph.add_edge(u, v, edge_index=int(index))
        return graph

    def test_pruning_validation_errors(self):
        seam_edges = {(0, 1), (1, 2)}
        view, _ = self._view_for_seam_edges(seam_edges)
        result = self._bridging_result(view, seam_edges)

        bad_shape = BridgingResult(
            bridged_edge_mask=np.zeros(view.edge_count + 1, dtype=bool),
            component_reports=tuple(),
            r_bridge=6,
        )
        with self.assertRaises(ValueError):
            compute_spur_pruning(view, bad_shape, anchor_boundary=False)

        bad_dtype = BridgingResult(
            bridged_edge_mask=result.bridged_edge_mask.astype(np.float64),
            component_reports=tuple(),
            r_bridge=6,
        )
        with self.assertRaises(ValueError):
            compute_spur_pruning(view, bad_dtype, anchor_boundary=False)
        with self.assertRaises(ValueError):
            compute_spur_pruning(view, result, l_min=0, anchor_boundary=False)
        with self.assertRaises(ValueError):
            compute_spur_pruning(view, result, l_min=4.5, anchor_boundary=False)
        with self.assertRaises(ValueError):
            compute_spur_pruning(view, result)
        with self.assertRaises(ValueError):
            compute_spur_pruning(view, result, anchor_boundary=False, extra_anchor_vertices=frozenset({-1}))

    def test_pruning_no_op_on_clean_long_chain(self):
        seam_edges = {(index, index + 1) for index in range(8)}
        view, _ = self._view_for_seam_edges(seam_edges)
        bridging = self._bridging_result(view, seam_edges)

        result = compute_spur_pruning(view, bridging, l_min=4, anchor_boundary=False)

        self.assertEqual(result.removed_edges, frozenset())
        self.assertTrue(np.array_equal(result.pruned_edge_mask, bridging.bridged_edge_mask))
        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.total_branches_pruned, 0)

    def test_pruning_removes_short_spur(self):
        main_chain = {(index, index + 1) for index in range(9)}
        spur = {(4, 10), (10, 11)}
        seam_edges = main_chain | spur
        view, _ = self._view_for_seam_edges(seam_edges)
        bridging = self._bridging_result(view, seam_edges)

        result, before, after = diagnose_pruning_application(view, bridging, l_min=4, anchor_boundary=False)

        self.assertEqual(self._mask_edges(view, result.pruned_edge_mask), main_chain)
        self.assertEqual(result.total_branches_pruned, 1)
        self.assertEqual(result.total_edges_removed, 2)
        self.assertGreaterEqual(before.junction_count, 1)
        self.assertEqual(after.junction_count, 0)

    def test_pruning_iterative_unmasking(self):
        seam_edges = {
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (3, 8),
            (3, 9),
            (6, 10),
        }
        view, _ = self._view_for_seam_edges(seam_edges)
        bridging = self._bridging_result(view, seam_edges)

        result = compute_spur_pruning(
            view,
            bridging,
            l_min=4,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({7}),
        )

        expected_survivors = {(3, 4), (4, 5), (5, 6), (6, 7)}
        expected_survivors.add((3, 9))
        self.assertEqual(result.total_iterations, 2)
        self.assertEqual(result.total_branches_pruned, 3)
        self.assertEqual(result.total_edges_removed, 5)
        self.assertEqual(self._mask_edges(view, result.pruned_edge_mask), expected_survivors)
        self.assertEqual([report['branches_pruned'] for report in result.iteration_reports], [3, 0])

    def test_pruning_protects_boundary_anchor_leaf(self):
        view, topology = self._grid_view()
        seam_edges = {(1, 5), (5, 6)}
        bridging = self._bridging_result(view, seam_edges)

        result = compute_spur_pruning(view, bridging, l_min=4, anchor_boundary=True, topology=topology)

        self.assertEqual(result.total_branches_pruned, 1)
        self.assertEqual(result.total_edges_removed, 2)
        self.assertFalse(np.any(result.pruned_edge_mask))

    def test_pruning_protects_boundary_anchor_root(self):
        view, topology = self._grid_view()
        seam_edges = {(0, 1), (1, 2), (1, 5), (5, 6)}
        bridging = self._bridging_result(view, seam_edges)

        result = compute_spur_pruning(view, bridging, l_min=4, anchor_boundary=True, topology=topology)
        graph = self._masked_graph(view, result.pruned_edge_mask)

        self.assertEqual(result.total_branches_pruned, 1)
        self.assertEqual(result.total_edges_removed, 2)
        self.assertEqual(int(graph.degree[1]), 2)

    def test_pruning_anchor_to_anchor_short_chain_preserved(self):
        view, topology = self._grid_view()
        seam_edges = {(0, 1), (1, 2)}
        bridging = self._bridging_result(view, seam_edges)

        result = compute_spur_pruning(view, bridging, l_min=4, anchor_boundary=True, topology=topology)

        self.assertEqual(result.total_branches_pruned, 0)
        self.assertEqual(result.total_edges_removed, 0)
        self.assertTrue(np.array_equal(result.pruned_edge_mask, bridging.bridged_edge_mask))

    def test_pruning_stick_component_both_unanchored(self):
        seam_edges = {(0, 1), (1, 2), (2, 3)}
        view, _ = self._view_for_seam_edges(seam_edges)
        bridging = self._bridging_result(view, seam_edges)

        result = compute_spur_pruning(view, bridging, l_min=4, anchor_boundary=False)

        self.assertEqual(result.total_branches_pruned, 1)
        self.assertEqual(result.total_edges_removed, 3)
        self.assertFalse(np.any(result.pruned_edge_mask))

    def test_pruning_cycle_no_leaves_unchanged(self):
        seam_edges = {(0, 1), (1, 2), (0, 2)}
        view, _ = self._view_for_seam_edges(seam_edges)
        bridging = self._bridging_result(view, seam_edges)

        result = compute_spur_pruning(view, bridging, l_min=4, anchor_boundary=False)

        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.total_branches_pruned, 0)
        self.assertTrue(np.array_equal(result.pruned_edge_mask, bridging.bridged_edge_mask))

    def test_pruning_lollipop_cuts_tail_keeps_loop(self):
        cycle = {(0, 1), (1, 2), (0, 2)}
        tail = {(0, 3), (3, 4)}
        view, _ = self._view_for_seam_edges(cycle | tail)
        bridging = self._bridging_result(view, cycle | tail)

        result = compute_spur_pruning(view, bridging, l_min=4, anchor_boundary=False)

        self.assertEqual(self._mask_edges(view, result.pruned_edge_mask), cycle)
        self.assertEqual(result.total_branches_pruned, 1)
        self.assertEqual(result.total_edges_removed, 2)

    def test_pruning_subset_invariant(self):
        rng = np.random.default_rng(12)
        topology, view = _strip_topology(length=6)
        mask = rng.random(view.edge_count) > 0.45
        bridging = BridgingResult(
            bridged_edge_mask=mask.astype(bool),
            component_reports=tuple(),
            r_bridge=6,
        )
        anchors = frozenset(int(vertex) for vertex in rng.choice(view.vertex_count, size=3, replace=False))

        result = compute_spur_pruning(
            view,
            bridging,
            l_min=4,
            anchor_boundary=True,
            extra_anchor_vertices=anchors,
            topology=topology,
        )

        self.assertTrue(np.all(result.pruned_edge_mask <= bridging.bridged_edge_mask))

    def test_pruning_branch_length_invariant(self):
        view, topology = self._grid_view()
        seam_edges = {(0, 1), (1, 2), (5, 6), (6, 7), (7, 11), (11, 15)}
        bridging = self._bridging_result(view, seam_edges)
        l_min = 4

        result = compute_spur_pruning(view, bridging, l_min=l_min, anchor_boundary=True, topology=topology)
        graph = self._masked_graph(view, result.pruned_edge_mask)
        anchors = set(boundary_vertices_from_topology(topology))

        for leaf in sorted(vertex for vertex, degree in graph.degree() if int(degree) == 1):
            previous = None
            current = int(leaf)
            length = 0
            endpoint = current
            while True:
                neighbors = [int(node) for node in graph.neighbors(current) if int(node) != previous]
                if not neighbors:
                    endpoint = current
                    break
                next_node = neighbors[0]
                length += 1
                endpoint = next_node
                if int(graph.degree[next_node]) != 2:
                    break
                previous = current
                current = next_node
            self.assertTrue(length >= l_min or int(leaf) in anchors or endpoint in anchors)

    def test_diagnose_pruning_application_round_trip(self):
        main_chain = {(index, index + 1) for index in range(9)}
        spur_a = {(4, 10), (10, 11)}
        spur_b = {(5, 12), (12, 13)}
        view, _ = self._view_for_seam_edges(main_chain | spur_a | spur_b)
        bridging = self._bridging_result(view, main_chain | spur_a | spur_b)

        pruning, before, after = diagnose_pruning_application(view, bridging, l_min=4, anchor_boundary=False)

        self.assertIsInstance(pruning, object)
        self.assertGreater(before.branch_count, after.branch_count)
        self.assertGreaterEqual(before.junction_count, after.junction_count)
        self.assertEqual(before.seam_edge_count - after.seam_edge_count, 4)


class PipelineTests(unittest.TestCase):
    @staticmethod
    def _masked_graph(view: SeamGraphView, mask: np.ndarray) -> nx.Graph:
        graph = nx.Graph()
        for index in np.flatnonzero(mask):
            u = int(view.unique_edges[index, 0])
            v = int(view.unique_edges[index, 1])
            graph.add_edge(u, v, edge_index=int(index))
        return graph

    @staticmethod
    def _assert_json_scalar_tree(test_case: unittest.TestCase, value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                test_case.assertIsInstance(key, str)
                PipelineTests._assert_json_scalar_tree(test_case, item)
            return
        if isinstance(value, list):
            for item in value:
                PipelineTests._assert_json_scalar_tree(test_case, item)
            return
        test_case.assertIsInstance(value, (int, float, str, bool, type(None)))

    def test_pipeline_validation_errors(self):
        _, view = _strip_topology(length=2)
        probabilities = np.zeros(view.edge_count, dtype=np.float64)

        with self.assertRaisesRegex(ValueError, 'anchor_boundary=True requires'):
            apply_topology_pipeline(view, probabilities, topology=None)

    def test_pipeline_end_to_end_thick_band_with_micro_gaps(self):
        view, topology = _grid_view(row_count=3, col_count=8)
        middle_row = 8
        probabilities = np.full(view.edge_count, 0.05, dtype=np.float64)
        middle_edges = [
            (middle_row + col, middle_row + col + 1)
            for col in range(7)
        ]
        middle_vertices = set(range(middle_row, middle_row + 8))
        for edge_index, edge in enumerate(view.unique_edges):
            if int(edge[0]) in middle_vertices or int(edge[1]) in middle_vertices:
                probabilities[edge_index] = 0.99
        probabilities[_edge_index(view, middle_edges[3])] = 0.40
        probabilities[_edge_index(view, middle_edges[4])] = 0.40

        result = apply_topology_pipeline(
            view,
            probabilities,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({middle_row, middle_row + 7}),
            topology=topology,
            max_bridge_euclidean_ratio=1.0,
        )

        self.assertGreater(result.skeleton_result.removals_committed, 0)
        self.assertGreaterEqual(result.bridging_result.bridges_accepted, 0)
        self.assertGreaterEqual(result.pruning_result.total_branches_pruned, 0)
        graph = self._masked_graph(view, result.final_edge_mask)
        self.assertTrue(nx.has_path(graph, middle_row, middle_row + 7))
        final_probabilities = np.where(result.final_edge_mask, 1.0, 0.0).astype(np.float64, copy=False)
        diagnostics = compute_seam_mask_diagnostics(view, final_probabilities, 0.5)
        self.assertEqual(diagnostics.thick_band_edge_count, 0)

    def test_pipeline_no_op_on_empty_probabilities(self):
        view, topology = _grid_view(row_count=3, col_count=4)
        probabilities = np.zeros(view.edge_count, dtype=np.float64)

        result = apply_topology_pipeline(
            view,
            probabilities,
            anchor_boundary=False,
            topology=topology,
        )

        self.assertFalse(np.any(result.final_edge_mask))
        self.assertEqual(result.skeleton_result.removals_committed, 0)
        self.assertEqual(result.bridging_result.bridges_accepted, 0)
        self.assertEqual(result.pruning_result.total_branches_pruned, 0)

    def test_pipeline_subset_of_initial_candidate(self):
        view, topology = _grid_view(row_count=3, col_count=6)
        probabilities = _edge_probability_vector(
            view,
            {
                (6, 7): 0.95,
                (7, 8): 0.95,
                (10, 11): 0.95,
                (8, 9): 0.35,
                (9, 10): 0.35,
            },
            default=0.05,
        )
        r_bridge = 2

        result = apply_topology_pipeline(
            view,
            probabilities,
            r_bridge=r_bridge,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({6, 11}),
            topology=topology,
        )

        allowed = set(result.skeleton_result.initial_candidate_vertices)
        queue = [(int(vertex), 0) for vertex in result.skeleton_result.skeleton_vertices]
        seen = {vertex for vertex, _ in queue}
        while queue:
            vertex, distance = queue.pop(0)
            allowed.add(vertex)
            if distance >= r_bridge:
                continue
            for neighbor in view.vertex_graph.neighbors(vertex):
                neighbor_index = int(neighbor)
                if neighbor_index in seen:
                    continue
                seen.add(neighbor_index)
                queue.append((neighbor_index, distance + 1))

        for edge_index in np.flatnonzero(result.final_edge_mask):
            u = int(view.unique_edges[edge_index, 0])
            v = int(view.unique_edges[edge_index, 1])
            self.assertIn(u, allowed)
            self.assertIn(v, allowed)

    def test_pipeline_telemetry_serialization(self):
        view, topology = _grid_view(row_count=3, col_count=4)
        probabilities = _edge_probability_vector(
            view,
            {
                (4, 5): 0.95,
                (5, 6): 0.95,
                (6, 7): 0.95,
            },
            default=0.05,
        )
        result = apply_topology_pipeline(
            view,
            probabilities,
            anchor_boundary=False,
            extra_anchor_vertices=frozenset({4, 7}),
            topology=topology,
        )

        payload = topology_pipeline_result_to_json_dict(result)
        encoded = json.dumps(payload, sort_keys=True)
        decoded = json.loads(encoded)

        self.assertEqual(set(payload), {'bridging', 'final_edge_count', 'parameters', 'pruning', 'skeleton'})
        self.assertEqual(
            set(payload['parameters']),
            {
                'tau_low',
                'd_max',
                'r_bridge',
                'l_min',
                'anchor_boundary',
                'max_bridge_euclidean_ratio',
                'max_debug_candidates',
                'max_endpoint_candidates',
                'min_loop_size_to_allow',
                'require_mutual_pairing',
                'tangent_alignment_weight',
            },
        )
        self.assertIn('component_reports', payload['bridging'])
        self.assertIn('accepted_bridge_reports', payload['bridging'])
        self.assertIn('iteration_reports', payload['pruning'])
        self.assertEqual(payload['final_edge_count'], int(result.final_edge_mask.sum()))
        self._assert_json_scalar_tree(self, payload)
        self.assertEqual(decoded, payload)


if __name__ == '__main__':
    unittest.main()
