import csv
import importlib.util
import io
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'tools' / 'evaluate_custom_dir.py'
spec = importlib.util.spec_from_file_location('evaluate_custom_dir_bridge', MODULE_PATH)
evaluate_custom_dir = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluate_custom_dir
spec.loader.exec_module(evaluate_custom_dir)

# Also import predict_seams the same way for monkeypatching
PS_PATH = ROOT / 'tools' / 'predict_seams.py'
ps_spec = importlib.util.spec_from_file_location('predict_seams_for_eval', PS_PATH)
predict_seams = importlib.util.module_from_spec(ps_spec)
sys.modules[ps_spec.name] = predict_seams
ps_spec.loader.exec_module(predict_seams)


def _payload(indices, vertex_count=4, edge_count=6, diagnostics=None):
    return {
        'topology': {
            'vertex_count': vertex_count,
            'edge_count': edge_count,
        },
        'seam_edge_indices': indices,
        'diagnostics': diagnostics or {},
    }


def _row(mesh_name='mesh.obj', **overrides):
    values = {
        'mesh_name': mesh_name,
        'mesh_path': str(Path(mesh_name).resolve()),
        'vertex_count': 4,
        'edge_count': 6,
        'v1_status': 'ok',
        'v1_error': None,
        'v1_seam_count': 2,
        'v1_time_s': 0.1,
        'v2_status': 'ok',
        'v2_error': None,
        'v2_seam_count': 2,
        'v2_time_s': 0.2,
        'jaccard': 1.0,
        'v1_only_count': 0,
        'v2_only_count': 0,
        'v2_skeleton_removals': 1,
        'v2_steiner_calls': 2,
        'v2_steiner_edges_added': 3,
        'v2_branches_pruned': 4,
        'v2_pruning_iterations': 5,
        'v2_thick_band_edges_after': 0,
    }
    values.update(overrides)
    return evaluate_custom_dir.MeshAblationRow(**values)


class EvaluateCustomDirTests(unittest.TestCase):
    def test_jaccard_basic(self):
        self.assertEqual(evaluate_custom_dir._jaccard({1, 2}, {1, 2}), 1.0)
        self.assertEqual(evaluate_custom_dir._jaccard({1}, {2}), 0.0)
        self.assertEqual(evaluate_custom_dir._jaccard({1, 2}, {2, 3}), 1.0 / 3.0)
        self.assertEqual(evaluate_custom_dir._jaccard(set(), set()), 1.0)

    def test_seam_indices_to_set_extracts_indices(self):
        payload = {'seam_edge_indices': [3, 1, 5, 1]}
        self.assertEqual(evaluate_custom_dir._seam_indices_to_set(payload), {1, 3, 5})

    def test_v2_telemetry_fields_with_v2_diagnostics(self):
        payload = {
            'diagnostics': {
                'postprocess_v2': {
                    'skeleton': {'removals_committed': 7},
                    'bridging': {'steiner_calls': 8, 'steiner_edges_added_total': 9},
                    'pruning': {'total_branches_pruned': 10, 'total_iterations': 11},
                }
            }
        }

        self.assertEqual(
            evaluate_custom_dir._v2_telemetry_fields(payload),
            {
                'skeleton_removals': 7,
                'steiner_calls': 8,
                'steiner_edges_added': 9,
                'branches_pruned': 10,
                'pruning_iterations': 11,
            },
        )

    def test_v2_telemetry_fields_without_v2_diagnostics(self):
        expected = {
            'skeleton_removals': None,
            'steiner_calls': None,
            'steiner_edges_added': None,
            'branches_pruned': None,
            'pruning_iterations': None,
        }
        self.assertEqual(evaluate_custom_dir._v2_telemetry_fields({'diagnostics': {}}), expected)
        self.assertEqual(evaluate_custom_dir._v2_telemetry_fields({}), expected)

    def test_evaluate_one_mesh_both_succeed(self):
        calls = []

        def fake_run_prediction(args):
            calls.append(args)
            if args.postprocess_version == 'v1':
                return _payload([1, 2, 3], edge_count=10)
            return _payload(
                [2, 3, 4],
                edge_count=10,
                diagnostics={
                    'seam_topology': {'thick_band_edge_count': 0},
                    'postprocess_v2': {
                        'skeleton': {'removals_committed': 1},
                        'bridging': {'steiner_calls': 2, 'steiner_edges_added_total': 3},
                        'pruning': {'total_branches_pruned': 4, 'total_iterations': 5},
                    },
                },
            )

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            evaluate_custom_dir.predict_seams,
            'run_prediction',
            side_effect=fake_run_prediction,
        ):
            row = evaluate_custom_dir.evaluate_one_mesh(
                Namespace(model_weights='weights.pt', write_all_edges=False),
                Path(tmp) / 'fake.obj',
                None,
                Path(tmp),
            )

        self.assertEqual(row.v1_status, 'ok')
        self.assertEqual(row.v2_status, 'ok')
        self.assertEqual(row.jaccard, 0.5)
        self.assertEqual(row.v1_only_count, 1)
        self.assertEqual(row.v2_only_count, 1)
        self.assertGreaterEqual(row.v1_time_s, 0.0)
        self.assertGreaterEqual(row.v2_time_s, 0.0)
        self.assertEqual([c.postprocess_version for c in calls], ['v1', 'v2'])
        self.assertNotEqual(calls[0].output_json, calls[1].output_json)

    def test_evaluate_one_mesh_v1_succeeds_v2_fails(self):
        def fake_run_prediction(args):
            if args.postprocess_version == 'v2':
                raise RuntimeError('synthetic v2 boom')
            return _payload([1, 2])

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            evaluate_custom_dir.predict_seams,
            'run_prediction',
            side_effect=fake_run_prediction,
        ):
            row = evaluate_custom_dir.evaluate_one_mesh(Namespace(), Path(tmp) / 'fake.obj', None, Path(tmp))

        self.assertEqual(row.v1_status, 'ok')
        self.assertEqual(row.v2_status, 'failed')
        self.assertIn('synthetic v2 boom', row.v2_error)
        self.assertEqual(row.jaccard, -1.0)
        self.assertEqual(row.v1_only_count, -1)
        self.assertEqual(row.v2_only_count, -1)
        self.assertEqual(row.v1_seam_count, 2)
        self.assertEqual(row.v2_seam_count, -1)

    def test_evaluate_one_mesh_v2_succeeds_v1_fails(self):
        def fake_run_prediction(args):
            if args.postprocess_version == 'v1':
                raise RuntimeError('synthetic v1 boom')
            return _payload([4, 5, 6])

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            evaluate_custom_dir.predict_seams,
            'run_prediction',
            side_effect=fake_run_prediction,
        ):
            row = evaluate_custom_dir.evaluate_one_mesh(Namespace(), Path(tmp) / 'fake.obj', None, Path(tmp))

        self.assertEqual(row.v1_status, 'failed')
        self.assertEqual(row.v2_status, 'ok')
        self.assertIn('synthetic v1 boom', row.v1_error)
        self.assertEqual(row.jaccard, -1.0)
        self.assertEqual(row.v1_only_count, -1)
        self.assertEqual(row.v2_only_count, -1)
        self.assertEqual(row.v1_seam_count, -1)
        self.assertEqual(row.v2_seam_count, 3)

    def test_evaluate_one_mesh_both_fail(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            evaluate_custom_dir.predict_seams,
            'run_prediction',
            side_effect=RuntimeError('boom'),
        ):
            row = evaluate_custom_dir.evaluate_one_mesh(Namespace(), Path(tmp) / 'fake.obj', None, Path(tmp))

        self.assertEqual(row.v1_status, 'failed')
        self.assertEqual(row.v2_status, 'failed')
        self.assertEqual(row.v1_seam_count, -1)
        self.assertEqual(row.v2_seam_count, -1)

    def test_discover_meshes_sorts_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['foo.obj', 'bar.obj', 'baz.obj', 'qux.txt']:
                (root / name).write_text('', encoding='utf-8')

            meshes = evaluate_custom_dir.discover_meshes(root, limit=2)

        self.assertEqual([p.name for p in meshes], ['bar.obj', 'baz.obj'])

    def test_discover_meshes_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(evaluate_custom_dir.discover_meshes(Path(tmp), limit=None), [])

    def test_discover_meshes_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            evaluate_custom_dir.discover_meshes(Path('does-not-exist-for-eval-test'), limit=None)

    def test_format_markdown_report_with_mixed_results(self):
        rows = [
            _row('ok.obj'),
            _row(
                'v2_failed.obj',
                v2_status='failed',
                v2_error='RuntimeError: nope',
                v2_seam_count=-1,
                v2_time_s=-1.0,
                jaccard=-1.0,
                v1_only_count=-1,
                v2_only_count=-1,
            ),
            _row(
                'both_failed.obj',
                v1_status='failed',
                v1_error='RuntimeError: v1',
                v1_seam_count=-1,
                v1_time_s=-1.0,
                v2_status='failed',
                v2_error='RuntimeError: v2',
                v2_seam_count=-1,
                v2_time_s=-1.0,
                jaccard=-1.0,
                v1_only_count=-1,
                v2_only_count=-1,
            ),
        ]

        report = evaluate_custom_dir.format_markdown_report(rows)

        self.assertTrue(report)
        self.assertIn('ok.obj', report)
        self.assertIn('v2_failed.obj', report)
        self.assertIn('both_failed.obj', report)
        self.assertIn('FAIL', report)
        self.assertIn('## Aggregate (both succeeded)', report)
        self.assertIn('## Failures', report)

    def test_write_csv_round_trip(self):
        rows = [
            _row('ok.obj'),
            _row(
                'failed.obj',
                v1_status='failed',
                v1_error='RuntimeError: boom',
                v1_seam_count=-1,
                v1_time_s=-1.0,
                v2_status='failed',
                v2_error='RuntimeError: boom',
                v2_seam_count=-1,
                v2_time_s=-1.0,
                jaccard=-1.0,
                v1_only_count=-1,
                v2_only_count=-1,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.csv'
            evaluate_custom_dir.write_csv(rows, path)
            with path.open('r', encoding='utf-8', newline='') as fh:
                records = list(csv.DictReader(fh))

        self.assertEqual(len(records), 2)
        self.assertIn('v2_thick_band_edges_after', records[0])
        self.assertEqual(records[1]['mesh_name'], 'failed.obj')
        self.assertEqual(records[1]['v1_seam_count'], '-1')
        self.assertEqual(records[1]['jaccard'], '-1.0')

    def test_main_with_no_obj_files_returns_5(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            evaluate_custom_dir,
            'discover_meshes',
            return_value=[],
        ):
            code = evaluate_custom_dir.main(
                ['--input-dir', tmp, '--model-weights', str(Path(tmp) / 'fake.pt'), '--quiet']
            )

        self.assertEqual(code, 5)

    def test_main_full_flow_two_meshes_two_versions(self):
        def fake_evaluate_one_mesh(base_args, mesh_path, keep_json_dir, tmp_dir):
            del base_args, keep_json_dir, tmp_dir
            return _row(mesh_path.name)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'foo.obj').write_text('', encoding='utf-8')
            (root / 'bar.obj').write_text('', encoding='utf-8')
            stdout = io.StringIO()
            with patch.object(evaluate_custom_dir, 'build_base_args', return_value=Namespace()), patch.object(
                evaluate_custom_dir,
                'evaluate_one_mesh',
                side_effect=fake_evaluate_one_mesh,
            ) as mock_eval, patch('sys.stdout', stdout):
                code = evaluate_custom_dir.main(
                    ['--input-dir', str(root), '--model-weights', str(root / 'fake.pt'), '--quiet']
                )

        self.assertEqual(code, 0)
        self.assertIn('bar.obj', stdout.getvalue())
        self.assertIn('foo.obj', stdout.getvalue())
        self.assertEqual(mock_eval.call_count, 2)


if __name__ == '__main__':
    unittest.main()
