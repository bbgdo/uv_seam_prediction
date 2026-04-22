import unittest

import numpy as np

from models.utils.postprocess import apply_seam_postprocessing, apply_seam_postprocessing_detailed


class PostprocessTests(unittest.TestCase):
    def test_skeletonization_thins_a_thick_band(self):
        unique_edges = np.asarray([
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (0, 3),
            (1, 4),
            (2, 5),
        ], dtype=np.int64)
        probabilities = np.asarray([0.90, 0.85, 0.88, 0.87, 0.80, 0.82, 0.79], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.5,
            max_gap_length=0,
            min_island_size=1,
        )

        self.assertTrue(np.array_equal(result.threshold_mask, np.ones(len(unique_edges), dtype=bool)))
        self.assertEqual(result.skeleton_deleted_vertices, (2, 3))
        self.assertTrue(np.array_equal(result.skeleton_mask, np.asarray([True, False, False, True, False, True, False])))

    def test_steiner_connects_disconnected_seam_segments(self):
        unique_edges = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
        probabilities = np.asarray([0.95, 0.45, 0.96], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.5,
            max_gap_length=5,
            min_island_size=1,
        )

        self.assertTrue(np.array_equal(result.skeleton_mask, np.asarray([True, False, True])))
        self.assertTrue(np.array_equal(result.steiner_mask, np.asarray([False, True, False])))
        self.assertEqual(result.steiner_added_edges, (1,))
        self.assertTrue(np.array_equal(result.final_mask, np.asarray([True, True, True])))

    def test_steiner_prefers_lower_cost_path(self):
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
            max_gap_length=5,
            min_island_size=1,
        )

        self.assertEqual(result.steiner_added_edges, (2, 3))
        self.assertTrue(result.final_mask[2])
        self.assertTrue(result.final_mask[3])
        self.assertFalse(result.final_mask[4])
        self.assertFalse(result.final_mask[5])

    def test_prunes_small_islands_after_skeleton_and_steiner(self):
        unique_edges = np.asarray([
            (0, 1),
            (1, 2),
            (2, 3),
            (5, 6),
            (6, 7),
        ], dtype=np.int64)
        probabilities = np.asarray([0.90, 0.92, 0.91, 0.88, 0.87], dtype=np.float64)

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

    def test_simple_api_returns_final_mask(self):
        unique_edges = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
        probabilities = np.asarray([0.95, 0.45, 0.96], dtype=np.float64)

        final_mask = apply_seam_postprocessing(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.5,
            max_gap_length=5,
            min_island_size=1,
        )

        self.assertTrue(np.array_equal(final_mask, np.asarray([True, True, True])))


if __name__ == '__main__':
    unittest.main()
