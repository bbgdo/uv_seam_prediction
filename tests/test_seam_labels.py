import unittest

from preprocessing.obj_parser import parse_obj_text
from preprocessing.seam_labels import extract_seam_truth
from preprocessing.topology import build_topology


NON_SEAM_SHARED_EDGE = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3
f 3/3 2/2 4/4
"""


SEAM_SHARED_EDGE = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0.2 0
vt 0.8 1
vt 0 1
f 1/1 2/2 3/3
f 3/5 2/4 4/6
"""


class SeamLabelTests(unittest.TestCase):
    def test_shared_edge_is_not_seam_when_uv_pairing_matches(self):
        topology = build_topology(parse_obj_text(NON_SEAM_SHARED_EDGE))
        truth = extract_seam_truth(topology)

        self.assertFalse(truth.seam_map[(1, 2)])
        self.assertFalse(truth.boundary_map[(1, 2)])

    def test_shared_edge_is_seam_when_uv_pairing_differs(self):
        topology = build_topology(parse_obj_text(SEAM_SHARED_EDGE))
        truth = extract_seam_truth(topology)

        self.assertTrue(truth.seam_map[(1, 2)])
        self.assertFalse(truth.boundary_map[(1, 2)])

    def test_boundary_edges_are_explicit(self):
        topology = build_topology(parse_obj_text(NON_SEAM_SHARED_EDGE))
        truth = extract_seam_truth(topology)

        boundary_edges = {edge for edge, is_boundary in truth.boundary_map.items() if is_boundary}

        self.assertEqual(boundary_edges, {(0, 1), (0, 2), (1, 3), (2, 3)})
        self.assertTrue(all(truth.seam_map[edge] for edge in boundary_edges))
        self.assertEqual(truth.audit.boundary_edges, 4)

    def test_opposite_face_direction_aligns_to_canonical_edge(self):
        topology = build_topology(parse_obj_text(NON_SEAM_SHARED_EDGE))
        truth = extract_seam_truth(topology)
        occurrences = topology.edge_incidence[(1, 2)]
        signatures = [
            truth.uv_signature_by_occurrence[(occ.face_index, occ.local_edge_index)]
            for occ in occurrences
        ]

        self.assertEqual(signatures, [(1, 2), (1, 2)])
        self.assertFalse(truth.seam_map[(1, 2)])


if __name__ == '__main__':
    unittest.main()
