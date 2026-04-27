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
MODULE_PATH = ROOT / 'tools' / 'evaluate_dir_topology.py'
spec = importlib.util.spec_from_file_location('evaluate_dir_topology_bridge', MODULE_PATH)
evaluate_dir_topology = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluate_dir_topology
spec.loader.exec_module(evaluate_dir_topology)


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
        'status': 'ok',
        'error': None,
        'seam_count': 2,
        'time_s': 0.2,
        'skeleton_removals': 1,
        'steiner_calls': 2,
        'steiner_edges_added': 3,
        'branches_pruned': 4,
        'pruning_iterations': 5,
        'thick_band_edges_after': 0,
    }
    values.update(overrides)
    return evaluate_dir_topology.MeshTopologyRow(**values)


class EvaluateDirTopologyTests(unittest.TestCase):
    def test_seam_indices_to_set_extracts_indices(self):
        payload = {'seam_edge_indices': [3, 1, 5, 1]}
        self.assertEqual(evaluate_dir_topology._seam_indices_to_set(payload), {1, 3, 5})

    def test_telemetry_fields_with_diagnostics(self):
        payload = {
            'diagnostics': {
                'postprocess': {
                    'skeleton': {'removals_committed': 7},
                    'bridging': {'steiner_calls': 8, 'steiner_edges_added_total': 9},
                    'pruning': {'total_branches_pruned': 10, 'total_iterations': 11},
                }
            }
        }

        self.assertEqual(
            evaluate_dir_topology._telemetry_fields(payload),
            {
                'skeleton_removals': 7,
                'steiner_calls': 8,
                'steiner_edges_added': 9,
                'branches_pruned': 10,
                'pruning_iterations': 11,
            },
        )

    def test_telemetry_fields_without_diagnostics(self):
        expected = {
            'skeleton_removals': None,
            'steiner_calls': None,
            'steiner_edges_added': None,
            'branches_pruned': None,
            'pruning_iterations': None,
        }
        self.assertEqual(evaluate_dir_topology._telemetry_fields({'diagnostics': {}}), expected)
        self.assertEqual(evaluate_dir_topology._telemetry_fields({}), expected)

    def test_evaluate_one_mesh_succeeds(self):
        calls = []

        def fake_run_prediction(args):
            calls.append(args)
            return _payload(
                [2, 3, 4],
                edge_count=10,
                diagnostics={
                    'seam_topology': {'thick_band_edge_count': 0},
                    'postprocess': {
                        'skeleton': {'removals_committed': 1},
                        'bridging': {'steiner_calls': 2, 'steiner_edges_added_total': 3},
                        'pruning': {'total_branches_pruned': 4, 'total_iterations': 5},
                    },
                },
            )

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            evaluate_dir_topology.predict_seams,
            'run_prediction',
            side_effect=fake_run_prediction,
        ):
            row = evaluate_dir_topology.evaluate_one_mesh(
                Namespace(model_weights='weights.pt', write_all_edges=False),
                Path(tmp) / 'fake.obj',
                None,
                Path(tmp),
            )

        self.assertEqual(row.status, 'ok')
        self.assertGreater(row.seam_count, 0)
        self.assertGreaterEqual(row.time_s, 0.0)
        self.assertEqual(row.steiner_edges_added, 3)
        self.assertEqual(len(calls), 1)

    def test_evaluate_one_mesh_fails(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            evaluate_dir_topology.predict_seams,
            'run_prediction',
            side_effect=RuntimeError('boom'),
        ):
            row = evaluate_dir_topology.evaluate_one_mesh(Namespace(), Path(tmp) / 'fake.obj', None, Path(tmp))

        self.assertEqual(row.status, 'failed')
        self.assertIn('boom', row.error)
        self.assertEqual(row.seam_count, -1)

    def test_discover_meshes_sorts_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['foo.obj', 'bar.obj', 'baz.obj', 'qux.txt']:
                (root / name).write_text('', encoding='utf-8')

            meshes = evaluate_dir_topology.discover_meshes(root, limit=2)

        self.assertEqual([p.name for p in meshes], ['bar.obj', 'baz.obj'])

    def test_discover_meshes_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(evaluate_dir_topology.discover_meshes(Path(tmp), limit=None), [])

    def test_discover_meshes_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            evaluate_dir_topology.discover_meshes(Path('does-not-exist-for-eval-test'), limit=None)

    def test_format_markdown_report_with_mixed_results(self):
        rows = [
            _row('ok.obj'),
            _row(
                'failed.obj',
                status='failed',
                error='RuntimeError: nope',
                seam_count=-1,
                time_s=-1.0,
            ),
        ]

        report = evaluate_dir_topology.format_markdown_report(rows)

        self.assertTrue(report)
        self.assertIn('ok.obj', report)
        self.assertIn('failed.obj', report)
        self.assertIn('| mesh | edges | seam count | time | skel removals | steiner edges | spurs pruned | thick after |', report)
        self.assertIn('FAIL', report)
        self.assertIn('## Aggregate', report)
        self.assertIn('## Failures', report)

    def test_write_csv_round_trip(self):
        rows = [
            _row('ok.obj'),
            _row(
                'failed.obj',
                status='failed',
                error='RuntimeError: boom',
                seam_count=-1,
                time_s=-1.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.csv'
            evaluate_dir_topology.write_csv(rows, path)
            with path.open('r', encoding='utf-8', newline='') as fh:
                records = list(csv.DictReader(fh))

        self.assertEqual(len(records), 2)
        self.assertIn('thick_band_edges_after', records[0])
        self.assertEqual(records[1]['mesh_name'], 'failed.obj')
        self.assertEqual(records[1]['seam_count'], '-1')
        self.assertIn('status', records[0])

    def test_main_with_no_obj_files_returns_5(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            evaluate_dir_topology,
            'discover_meshes',
            return_value=[],
        ):
            code = evaluate_dir_topology.main(
                ['--input-dir', tmp, '--model-weights', str(Path(tmp) / 'fake.pt'), '--quiet']
            )

        self.assertEqual(code, 5)

    def test_main_full_flow_two_meshes(self):
        def fake_evaluate_one_mesh(base_args, mesh_path, keep_json_dir, tmp_dir):
            del base_args, keep_json_dir, tmp_dir
            return _row(mesh_path.name)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'foo.obj').write_text('', encoding='utf-8')
            (root / 'bar.obj').write_text('', encoding='utf-8')
            stdout = io.StringIO()
            with patch.object(evaluate_dir_topology, 'build_base_args', return_value=Namespace()), patch.object(
                evaluate_dir_topology,
                'evaluate_one_mesh',
                side_effect=fake_evaluate_one_mesh,
            ) as mock_eval, patch('sys.stdout', stdout):
                code = evaluate_dir_topology.main(
                    ['--input-dir', str(root), '--model-weights', str(root / 'fake.pt'), '--quiet']
                )

        self.assertEqual(code, 0)
        self.assertIn('bar.obj', stdout.getvalue())
        self.assertIn('foo.obj', stdout.getvalue())
        self.assertEqual(mock_eval.call_count, 2)


if __name__ == '__main__':
    unittest.main()
