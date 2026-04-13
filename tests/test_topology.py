import unittest

from preprocessing.obj_parser import parse_obj_text
from preprocessing.topology import TopologyError, WeldConfig, build_topology, canonical_edge_key


UV_SEAM_FIXTURE = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 0.2
vt 1 0.8
vt 0 1
f 1/1 2/2 3/3
f 1/4 3/5 4/6
"""


class TopologyTests(unittest.TestCase):
    def test_builds_stable_edge_keys_and_incidence(self):
        mesh = parse_obj_text(UV_SEAM_FIXTURE)
        topology = build_topology(mesh)

        self.assertEqual(canonical_edge_key(3, 1), (1, 3))
        self.assertEqual(len(topology.canonical_edges), 5)
        self.assertIn((0, 2), topology.edge_incidence)
        self.assertEqual(len(topology.edge_incidence[(0, 2)]), 2)
        self.assertEqual(len(topology.edge_incidence[(0, 1)]), 1)
        self.assertEqual(topology.edge_coordinates[(0, 1)], ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    def test_exact_mode_preserves_original_vertex_identity(self):
        mesh = parse_obj_text(UV_SEAM_FIXTURE)
        topology = build_topology(mesh, WeldConfig.exact())

        self.assertEqual(topology.original_vertex_to_canonical_gid, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual(topology.canonical_gid_to_original_vertex, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual(topology.weld_audit.welded_vertex_count, 0)

    def test_preserves_corner_provenance_for_future_seam_labeling(self):
        mesh = parse_obj_text(UV_SEAM_FIXTURE)
        topology = build_topology(mesh)

        occurrences = topology.edge_incidence[(0, 2)]
        uv_pairs = [
            (occ.corner_a.uv_index, occ.corner_b.uv_index)
            for occ in occurrences
        ]

        self.assertEqual(uv_pairs, [(2, 0), (3, 4)])
        self.assertEqual([occ.face_line_number for occ in occurrences], [12, 13])

    def test_welded_mode_is_explicit_and_audited(self):
        mesh = parse_obj_text(
            """
            v 0 0 0
            v 1 0 0
            v 0 1 0
            v 1.0001 0 0
            v 1 1 0
            f 1 2 3
            f 4 5 3
            """
        )

        exact_topology = build_topology(mesh)
        welded_topology = build_topology(mesh, WeldConfig.welded(quantization=0.01))

        self.assertEqual(len(exact_topology.canonical_vertices), 5)
        self.assertEqual(len(welded_topology.canonical_vertices), 4)
        self.assertEqual(welded_topology.original_vertex_to_canonical_gid[1], 1)
        self.assertEqual(welded_topology.original_vertex_to_canonical_gid[3], 1)
        self.assertEqual(welded_topology.weld_audit.welded_vertex_count, 1)
        self.assertEqual(welded_topology.weld_audit.weld_groups[1], (1, 3))

    def test_rejects_non_manifold_edges(self):
        mesh = parse_obj_text(
            """
            v 0 0 0
            v 1 0 0
            v 0 1 0
            v 0 -1 0
            v 0 0 1
            f 1 2 3
            f 2 1 4
            f 1 2 5
            """
        )

        with self.assertRaisesRegex(TopologyError, 'non-manifold edge'):
            build_topology(mesh)

    def test_rejects_degenerate_welded_faces(self):
        mesh = parse_obj_text(
            """
            v 0 0 0
            v 0.0001 0 0
            v 0 1 0
            f 1 2 3
            """
        )

        with self.assertRaisesRegex(TopologyError, 'degenerate triangle'):
            build_topology(mesh, WeldConfig.welded(quantization=0.01))


if __name__ == '__main__':
    unittest.main()
