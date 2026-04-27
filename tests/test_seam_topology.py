import unittest
from types import SimpleNamespace

import numpy as np

from models.utils.seam_topology import (
    build_seam_graph_view,
    compute_seam_mask_diagnostics,
    diagnostics_to_json_dict,
)
from preprocessing.obj_parser import ObjCorner, ObjFace, ObjMesh
from preprocessing.topology import WeldConfig, build_topology


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


if __name__ == '__main__':
    unittest.main()
