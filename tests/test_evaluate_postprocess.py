import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from preprocessing.obj_parser import ObjParseError


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'tools' / 'evaluate_postprocess.py'
spec = importlib.util.spec_from_file_location('evaluate_postprocess_under_test', MODULE_PATH)
evaluate_postprocess = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluate_postprocess
spec.loader.exec_module(evaluate_postprocess)


def _sample(file_path: str):
    return SimpleNamespace(file_path=file_path)


class EvaluatePostprocessTests(unittest.TestCase):
    def test_select_test_samples_uses_split_filter_and_limit(self):
        dataset = [
            _sample('mesh_keep_0.obj'),
            _sample('mesh_drop_0.obj'),
            _sample('mesh_keep_1.obj'),
        ]
        split_info = {'group_mode': 'legacy', 'test': ['mesh_keep_0', 'mesh_keep_1']}

        selected = evaluate_postprocess._select_test_samples(dataset, split_info, limit_meshes=1)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].file_path, 'mesh_keep_0.obj')

    def test_evaluate_checkpoint_bypasses_dataset_loader_for_standalone_mesh(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            mesh_path = tmp / 'standalone.obj'
            weights_path = tmp / 'best_model.pth'
            config_path = tmp / 'config.json'
            summary_path = tmp / 'summary.json'
            for path in (mesh_path, weights_path, config_path, summary_path):
                path.write_text('{}\n', encoding='utf-8')

            args = Namespace(
                model_weights=str(weights_path),
                config_json=str(config_path),
                summary_json=str(summary_path),
                dataset_path=None,
                threshold=0.6,
                max_gap_length=3,
                min_island_size=3,
                device='cpu',
                limit_meshes=None,
                mesh_path=str(mesh_path),
                debug_export=True,
                output_json=None,
                model_type='auto',
            )

            stub_report = {'status': 'completed', 'split': {'evaluated_mesh_count': 1}}

            with (
                patch.object(evaluate_postprocess.predict_bridge, 'load_json', side_effect=[{}, {}]),
                patch.object(evaluate_postprocess.predict_bridge, 'resolve_model_type', return_value='graphsage'),
                patch.object(evaluate_postprocess.predict_bridge, 'resolve_device', return_value='cpu'),
                patch.object(evaluate_postprocess, '_resolve_model_kwargs_for_evaluation', return_value={'in_dim': 14}),
                patch.object(evaluate_postprocess, '_evaluate_arbitrary_mesh', return_value=stub_report) as eval_mesh,
                patch.object(evaluate_postprocess, '_load_dataset_for_run', side_effect=AssertionError('dataset loader should not be used')),
            ):
                report = evaluate_postprocess.evaluate_checkpoint(args)

            self.assertIs(report, stub_report)
            eval_mesh.assert_called_once()
            self.assertEqual(eval_mesh.call_args.kwargs['mesh_path'], mesh_path.resolve())

    def test_load_topology_for_standalone_mesh_can_triangulate_quad_obj(self):
        quad_obj = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = Path(tmp_dir) / 'quad.obj'
            mesh_path.write_text(quad_obj, encoding='utf-8')

            with self.assertRaises(ObjParseError):
                evaluate_postprocess._load_topology_for_standalone_mesh(mesh_path, triangulate=False)

            topology, unique_edges = evaluate_postprocess._load_topology_for_standalone_mesh(
                mesh_path,
                triangulate=True,
            )

            self.assertEqual(len(topology.canonical_faces), 2)
            self.assertEqual(unique_edges.shape[1], 2)
            self.assertGreaterEqual(len(unique_edges), 5)


if __name__ == '__main__':
    unittest.main()
