import unittest

import numpy as np

from models.utils.postprocess import apply_seam_postprocessing, apply_seam_postprocessing_detailed


class PostprocessTests(unittest.TestCase):
    def test_hysteresis_removes_isolated_weak_noise(self):
        unique_edges = np.asarray([(0, 1), (2, 3)], dtype=np.int64)
        probabilities = np.asarray([0.45, 0.70], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.60,
            max_gap_length=0,
            min_island_size=1,
            smoothing_iterations=0,
        )

        self.assertTrue(np.array_equal(result.final_mask, np.asarray([False, True])))

    def test_hysteresis_keeps_weak_edges_connected_to_strong_edges(self):
        unique_edges = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
        probabilities = np.asarray([0.70, 0.45, 0.72], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.60,
            max_gap_length=0,
            min_island_size=1,
            smoothing_iterations=0,
        )

        self.assertTrue(np.array_equal(result.final_mask, np.asarray([True, True, True])))

    def test_spur_pruning_removes_low_confidence_t_branch(self):
        unique_edges = np.asarray([(0, 1), (1, 2), (2, 3), (2, 4)], dtype=np.int64)
        probabilities = np.asarray([0.80, 0.80, 0.80, 0.45], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.60,
            max_gap_length=0,
            min_island_size=1,
            smoothing_iterations=0,
        )

        self.assertTrue(np.array_equal(result.final_mask, np.asarray([True, True, True, False])))
        self.assertEqual(result.pruned_edge_indices, (3,))

    def test_spur_pruning_preserves_simple_open_path(self):
        unique_edges = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
        probabilities = np.asarray([0.70, 0.46, 0.70], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.60,
            max_gap_length=0,
            min_island_size=1,
            smoothing_iterations=0,
        )

        self.assertTrue(np.array_equal(result.final_mask, np.asarray([True, True, True])))
        self.assertEqual(result.pruned_edge_indices, ())

    def test_local_bridge_closes_bounded_below_low_gap(self):
        unique_edges = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
        probabilities = np.asarray([0.70, 0.50, 0.70], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.60,
            tau_low=0.55,
            max_gap_length=1,
            min_island_size=1,
            smoothing_iterations=0,
        )

        self.assertEqual(result.steiner_added_edges, (1,))
        self.assertTrue(np.array_equal(result.final_mask, np.asarray([True, True, True])))

    def test_simple_api_returns_final_mask(self):
        unique_edges = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
        probabilities = np.asarray([0.70, 0.45, 0.72], dtype=np.float64)

        final_mask = apply_seam_postprocessing(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.60,
            max_gap_length=0,
            min_island_size=1,
            smoothing_iterations=0,
        )

        self.assertTrue(np.array_equal(final_mask, np.asarray([True, True, True])))


if __name__ == '__main__':
    unittest.main()
