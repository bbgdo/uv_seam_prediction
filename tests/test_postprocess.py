import unittest

import numpy as np

from models.utils.postprocess import (
    apply_seam_postprocessing,
    apply_seam_postprocessing_detailed,
)


class PostprocessTests(unittest.TestCase):
    def test_closes_single_edge_gap_between_terminals(self):
        unique_edges = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
        probabilities = np.asarray([0.95, 0.45, 0.96], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.5,
            max_gap_length=2,
            min_island_size=1,
        )

        self.assertEqual(result.closed_paths, ((1,),))
        self.assertEqual(result.added_edge_indices, (1,))
        self.assertTrue(np.array_equal(result.final_mask, np.asarray([True, True, True])))

    def test_respects_gap_length_limit(self):
        unique_edges = np.asarray([(0, 1), (1, 2), (2, 3), (3, 4)], dtype=np.int64)
        probabilities = np.asarray([0.95, 0.45, 0.44, 0.96], dtype=np.float64)

        mask = apply_seam_postprocessing(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.5,
            max_gap_length=1,
            min_island_size=1,
        )

        self.assertTrue(np.array_equal(mask, np.asarray([True, False, False, True])))

    def test_prefers_low_cost_high_probability_bridge(self):
        unique_edges = np.asarray([
            (0, 1),
            (3, 4),
            (1, 2),
            (2, 3),
            (1, 5),
            (5, 3),
        ], dtype=np.int64)
        probabilities = np.asarray([0.98, 0.99, 0.49, 0.48, 0.20, 0.19], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.5,
            max_gap_length=3,
            min_island_size=1,
        )

        self.assertEqual(result.closed_paths, ((2, 3),))
        self.assertEqual(result.added_edge_indices, (2, 3))
        self.assertTrue(result.final_mask[2])
        self.assertTrue(result.final_mask[3])
        self.assertFalse(result.final_mask[4])
        self.assertFalse(result.final_mask[5])

    def test_prunes_small_islands_after_gap_closing(self):
        unique_edges = np.asarray([
            (0, 1),
            (1, 2),
            (2, 3),
            (5, 6),
            (6, 7),
        ], dtype=np.int64)
        probabilities = np.asarray([0.9, 0.92, 0.91, 0.88, 0.87], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.5,
            max_gap_length=0,
            min_island_size=3,
        )

        self.assertEqual(result.pruned_component_count, 1)
        self.assertEqual(result.pruned_edge_indices, (3, 4))
        self.assertTrue(np.array_equal(result.final_mask, np.asarray([True, True, True, False, False])))


if __name__ == '__main__':
    unittest.main()
