import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from models.utils.postprocess import apply_seam_postprocessing, apply_seam_postprocessing_detailed


class PostprocessTests(unittest.TestCase):
    def test_self_bridge_closes_broken_loop_before_cross_bridge(self):
        unique_edges = np.asarray([
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 4),
            (4, 3),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
        ], dtype=np.int64)
        probabilities = np.asarray([0.9, 0.88, 0.87, 0.7, 0.69, 0.95, 0.94, 0.93, 0.92], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.8,
            r_self=2,
            r_cross=2,
        )

        self.assertTrue(result.final_mask[3])
        self.assertTrue(result.final_mask[4])
        self.assertEqual(set(result.steiner_added_edges), {3, 4})

    def test_cross_bridge_attaches_fragment_to_main_graph(self):
        unique_edges = np.asarray([
            (0, 1),
            (1, 2),
            (4, 5),
            (2, 3),
            (3, 4),
        ], dtype=np.int64)
        probabilities = np.asarray([0.95, 0.94, 0.90, 0.12, 0.11], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.8,
            r_self=1,
            r_cross=3,
            ambiguity_margin=0.0,
            force_close_max_edges=3,
        )

        self.assertTrue(np.array_equal(result.final_mask, np.ones(5, dtype=bool)))
        self.assertEqual(set(result.steiner_added_edges), {3, 4})

    def test_self_bridge_rejects_protected_third_party_path(self):
        unique_edges = np.asarray([
            (0, 1),
            (3, 4),
            (2, 3),
            (4, 5),
            (5, 0),
            (1, 2),
        ], dtype=np.int64)
        probabilities = np.asarray([0.95, 0.96, 0.80, 0.80, 0.80, 0.94], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.9,
            r_self=4,
            r_cross=0,
            garbage_max_edges=0,
            protect_min_edges=1,
        )

        self.assertTrue(np.array_equal(result.final_mask, np.asarray([True, True, False, False, False, True])))
        self.assertEqual(set(result.steiner_added_edges), set())

    def test_aggressive_garbage_collection_deletes_small_open_fragment(self):
        unique_edges = np.asarray([
            (0, 1),
            (1, 2),
            (2, 3),
            (10, 11),
            (11, 12),
        ], dtype=np.int64)
        probabilities = np.asarray([0.95, 0.94, 0.93, 0.8, 0.79], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.5,
            r_cross=1,
            garbage_max_edges=4,
        )

        self.assertTrue(np.array_equal(result.final_mask, np.asarray([True, True, True, False, False])))
        self.assertEqual(set(result.pruned_edge_indices), {3, 4})

    def test_band_collapse_removes_near_main_satellite(self):
        unique_edges = np.asarray([
            (0, 1),
            (1, 2),
            (2, 3),
            (5, 6),
            (6, 7),
            (5, 7),
            (1, 5),
            (3, 7),
        ], dtype=np.int64)
        probabilities = np.asarray([0.95, 0.94, 0.93, 0.80, 0.79, 0.78, 0.35, 0.34], dtype=np.float64)

        result = apply_seam_postprocessing_detailed(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.5,
            r_cross=2,
            r_snap=3,
            snap_max_edges=6,
            r_band=2,
        )

        self.assertTrue(np.array_equal(result.final_mask, np.asarray([True, True, True, False, False, False, True, True])))

    def test_simple_api_returns_binary_mask(self):
        unique_edges = np.asarray([
            (0, 1),
            (1, 2),
            (2, 3),
            (10, 11),
            (11, 12),
        ], dtype=np.int64)
        probabilities = np.asarray([0.95, 0.94, 0.93, 0.8, 0.79], dtype=np.float64)

        final_mask = apply_seam_postprocessing(
            topology=None,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=0.5,
            r_cross=1,
        )

        self.assertEqual(final_mask.dtype, np.bool_)
        self.assertTrue(np.array_equal(final_mask, np.asarray([True, True, True, False, False])))

    def test_debug_export_writes_bridge_artifacts(self):
        unique_edges = np.asarray([
            (0, 1),
            (1, 2),
            (4, 5),
            (2, 3),
            (3, 4),
        ], dtype=np.int64)
        probabilities = np.asarray([0.95, 0.94, 0.90, 0.70, 0.69], dtype=np.float64)
        topology = type('Topology', (), {
            'canonical_vertices': np.asarray([
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (5.0, 0.0, 0.0),
            ], dtype=np.float64),
            'canonical_edges': unique_edges,
        })()

        with TemporaryDirectory() as tmp_dir:
            result = apply_seam_postprocessing_detailed(
                topology=topology,
                unique_edges=unique_edges,
                probabilities=probabilities,
                threshold=0.8,
                r_self=1,
                r_cross=3,
                ambiguity_margin=0.0,
                debug_export_dir=tmp_dir,
            )

            export_dir = Path(tmp_dir)
            terminals_obj = export_dir / 'terminals.obj'
            candidates_json = export_dir / 'bridge_candidates.json'
            accepted_obj = export_dir / 'accepted_bridges.obj'

            self.assertTrue(result.final_mask.dtype == np.bool_)
            self.assertTrue(terminals_obj.exists())
            self.assertTrue(candidates_json.exists())
            self.assertTrue(accepted_obj.exists())
            self.assertIn('p ', terminals_obj.read_text(encoding='utf-8'))
            self.assertIn('l ', accepted_obj.read_text(encoding='utf-8'))

            payload = json.loads(candidates_json.read_text(encoding='utf-8'))
            self.assertIn('stage_b', payload)
            self.assertIn('stage_c', payload)
            self.assertIn('top_rejected_bridges', payload)
            self.assertIn('accepted_bridges', payload)


if __name__ == '__main__':
    unittest.main()
