import unittest
from types import SimpleNamespace

import networkx as nx
import numpy as np

from models.utils.seam_topology import (
    SeamGraphView,
    build_seam_graph_view,
    boundary_vertices_from_topology,
    compute_topology_preserving_skeleton,
    compute_seam_mask_diagnostics,
    diagnose_skeleton_application,
    diagnostics_to_json_dict,
    lift_edge_probabilities_to_vertices,
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
        """
        REGRESSION GUARD for the tau_high=0.70 anchor bug.
        Under the original implementation, a thick band with all
        probabilities >= 0.95 was frozen because every vertex was an anchor.
        Under the corrected implementation, anchors are structural, so the
        band is thinned regardless of probability magnitude.
        """
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


if __name__ == '__main__':
    unittest.main()
