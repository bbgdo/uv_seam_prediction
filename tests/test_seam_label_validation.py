import unittest

import numpy as np

from tools.validate_seam_labels import audit_vertex_remap, compare_seams


class SeamLabelValidationTests(unittest.TestCase):
    def test_remap_reports_many_to_one_without_error(self):
        split = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        merged = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        audit = audit_vertex_remap(split, merged)

        self.assertEqual(audit['many_to_one_vertices'], 1)
        self.assertEqual(audit['many_to_one_split_vertices'], 2)
        self.assertEqual(audit['nonzero_reconstruction_errors'], 0)
        self.assertFalse(audit['suspicious'])

    def test_compare_seams_counts_fp_and_fn(self):
        pipeline = {(0, 1), (1, 2)}
        reference = {(1, 2), (2, 3)}

        metrics = compare_seams(pipeline, reference, edge_count=4)

        self.assertEqual(metrics['tp'], 1)
        self.assertEqual(metrics['fp'], 1)
        self.assertEqual(metrics['fn'], 1)
        self.assertEqual(metrics['mismatch_count'], 2)
        self.assertAlmostEqual(metrics['precision'], 0.5)
        self.assertAlmostEqual(metrics['recall'], 0.5)


if __name__ == '__main__':
    unittest.main()
