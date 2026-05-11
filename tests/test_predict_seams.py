import importlib.util
import contextlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'tools' / 'predict_seams.py'
spec = importlib.util.spec_from_file_location('predict_seams_bridge', MODULE_PATH)
predict_seams = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = predict_seams
spec.loader.exec_module(predict_seams)

from preprocessing.build_gnn_dataset import build_dual_edge_index_from_unique_edges  # noqa: E402
from preprocessing.obj_parser import parse_obj_text  # noqa: E402
from preprocessing.topology import WeldConfig, build_topology  # noqa: E402


SQUARE_OBJ = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3
f 1 3 4
"""


def _args(feature_bundle='paper14', **overrides):
    values = {
        'feature_bundle': feature_bundle,
        'enable_ao': False,
        'enable_dihedral': False,
        'enable_symmetry': False,
        'enable_density': False,
        'enable_thickness_sdf': False,
    }
    values.update(overrides)
    return Namespace(**values)


def _square_topology():
    return build_topology(parse_obj_text(SQUARE_OBJ), WeldConfig.exact())


class _DummyModel(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self._logits = [
            float(np.log(float(probability) / (1.0 - float(probability))))
            for probability in probabilities
        ]

    def forward(self, x, edge_index):
        del edge_index
        return torch.asarray(self._logits[:x.shape[0]], dtype=torch.float32, device=x.device)


class PredictSeamsTests(unittest.TestCase):
    def test_threshold_resolution_precedence(self):
        summary = {'best_validation_threshold': 0.7}

        self.assertEqual(predict_seams.resolve_threshold(0.8, summary), 0.8)
        self.assertEqual(predict_seams.resolve_threshold(None, summary), 0.7)
        with self.assertRaisesRegex(predict_seams.PredictionError, 'threshold is required'):
            predict_seams.resolve_threshold(None, {})

    def test_parser_rejects_threshold_policy_toggle(self):
        with self.assertRaises(SystemExit):
            predict_seams.parse_args([
                '--mesh-path', 'mesh.obj',
                '--model-weights', 'weights.pt',
                '--output-json', 'out.json',
                '--no-fail-if-threshold-missing',
            ])

    def test_model_type_resolution_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / 'gatv2_run' / 'best_model.pth'

            self.assertEqual(predict_seams.resolve_model_type('graphsage', {'model': 'gatv2'}, weights), 'graphsage')
            self.assertEqual(predict_seams.resolve_model_type('auto', {'model_name': 'DualGraphSAGE'}, weights), 'graphsage')
            self.assertEqual(predict_seams.resolve_model_type('auto', {'model': 'DualGATv2'}, weights), 'gatv2')
            self.assertEqual(predict_seams.resolve_model_type('sparsemeshcnn', {}, weights), 'sparsemeshcnn')

            with self.assertRaisesRegex(predict_seams.PredictionError, 'unsupported model type'):
                predict_seams.resolve_model_type('unknown', {}, weights)
            with self.assertRaisesRegex(predict_seams.PredictionError, 'model type could not be resolved'):
                predict_seams.resolve_model_type('auto', {}, weights)
            with self.assertRaisesRegex(predict_seams.PredictionError, 'model type could not be resolved'):
                predict_seams.resolve_model_type('auto', {'model_name': 'my_gatv2_experiment'}, weights)
            self.assertEqual(
                predict_seams.resolve_model_type('auto', {'model': 'meshcnn_full'}, Path(tmp) / 'run' / 'best_model.pth'),
                'sparsemeshcnn',
            )

    def test_cli_rejects_meshcnn_full_as_model_type(self):
        with self.assertRaises(SystemExit):
            predict_seams._normalize_cli_model_type('meshcnn_full')
        with self.assertRaises(SystemExit):
            predict_seams._normalize_cli_model_type('meshcnn')
        with self.assertRaises(SystemExit):
            predict_seams._normalize_cli_model_type('sparse_meshcnn')

    def test_cli_accepts_sparsemeshcnn_model_type(self):
        self.assertEqual(predict_seams._normalize_cli_model_type('sparsemeshcnn'), 'sparsemeshcnn')
        self.assertEqual(predict_seams._normalize_cli_model_type('gatv2'), 'gatv2')
        self.assertEqual(predict_seams._normalize_cli_model_type('graphsage'), 'graphsage')

    def test_feature_bundle_resolution(self):
        selection, endpoint_order, _ = predict_seams.resolve_feature_bundle(_args('paper14'), {}, {})
        self.assertEqual(selection.feature_group, 'paper14')
        self.assertEqual(selection.feature_count, 14)
        self.assertEqual(endpoint_order, 'random')

        selection, endpoint_order, _ = predict_seams.resolve_feature_bundle(
            _args('paper14'),
            {'endpoint_order': 'fixed'},
            {},
        )
        self.assertEqual(selection.feature_group, 'paper14')
        self.assertEqual(endpoint_order, 'fixed')

        selection, endpoint_order, _ = predict_seams.resolve_feature_bundle(
            _args('custom', enable_ao=True, enable_density=True),
            {},
            {},
        )
        self.assertEqual(selection.feature_group, 'custom')
        self.assertTrue(selection.feature_flags.ao)
        self.assertTrue(selection.feature_flags.density)
        self.assertEqual(endpoint_order, 'fixed')

        with self.assertRaisesRegex(predict_seams.PredictionError, 'requires at least one'):
            predict_seams.resolve_feature_bundle(_args('custom'), {}, {})

    def test_canonical_edge_order_mismatch_raises(self):
        topology = _square_topology()
        wrong_edges = np.asarray([(0, 2), (0, 1), (0, 3), (1, 2), (2, 3)], dtype=np.int64)

        with self.assertRaisesRegex(predict_seams.PredictionError, 'canonical edge order mismatch'):
            predict_seams.assert_canonical_edge_order(wrong_edges, topology.canonical_edges, Path('mesh.obj'))

    def test_json_payload_semantics(self):
        topology = _square_topology()
        unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64)
        probabilities = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3], dtype=np.float32)
        seam_mask = probabilities >= 0.75
        selection, _, _ = predict_seams.resolve_feature_bundle(_args('paper14'), {}, {})

        payload = predict_seams.build_output_payload(
            mesh_path=Path('mesh.obj'),
            output_json=Path('out.json'),
            weights_path=Path('best_model.pth'),
            config_path=Path('config.json'),
            summary_path=Path('summary.json'),
            model_type='gatv2',
            feature_bundle='paper14',
            selection=selection,
            threshold=0.75,
            device=torch.device('cpu'),
            topology=topology,
            unique_edges=unique_edges,
            probabilities=probabilities,
            seam_mask=seam_mask,
            write_all_edges=True,
        )

        self.assertEqual(payload['seam_edge_indices'], [1, 3])
        self.assertEqual([row['canonical_edge_index'] for row in payload['seam_edges']], [1, 3])
        self.assertEqual([row['canonical_edge_index'] for row in payload['edges']], [0, 1, 2, 3, 4])
        self.assertEqual(payload['edges'][0]['vertex_ids_0based'], [0, 1])
        self.assertEqual(payload['edges'][0]['vertex_ids_obj_1based'], [1, 2])
        self.assertEqual(payload['seam_edges'][0]['vertex_ids_0based'], [0, 2])
        self.assertEqual(payload['seam_edges'][0]['vertex_ids_obj_1based'], [1, 3])

    def test_meshcnn_inference_sample_is_unlabeled(self):
        topology = _square_topology()
        feature_mesh = predict_seams.build_feature_mesh_from_canonical_topology(topology)
        selection, endpoint_order, _ = predict_seams.resolve_feature_bundle(_args('paper14'), {}, {})
        edge_features = np.zeros((len(topology.canonical_edges), selection.feature_count), dtype=np.float32)
        unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64)

        sample = predict_seams.build_meshcnn_inference_sample(
            mesh_path=Path('mesh.obj'),
            feature_mesh=feature_mesh,
            unique_edges=unique_edges,
            edge_features=edge_features,
            selection=selection,
            endpoint_order=endpoint_order,
            topology=topology,
        )

        self.assertEqual(sample.label_source, 'inference_unlabeled')
        self.assertTrue(torch.equal(sample.edge_labels, torch.zeros(len(unique_edges))))
        self.assertEqual(sample.edge_features.shape, edge_features.shape)
        self.assertEqual(sample.unique_edges.tolist(), unique_edges.tolist())

    def test_write_all_edges_includes_full_table(self):
        payload = self._payload(write_all_edges=True)

        self.assertIn('edges', payload)
        self.assertEqual(len(payload['edges']), 5)

    def test_no_write_all_edges_omits_full_table_but_keeps_metadata(self):
        payload = self._payload(write_all_edges=False)

        self.assertNotIn('edges', payload)
        self.assertIn('model', payload)
        self.assertIn('topology', payload)
        self.assertEqual(payload['seam_edge_indices'], [1, 3])
        self.assertEqual([row['canonical_edge_index'] for row in payload['seam_edges']], [1, 3])

    def test_error_path_writes_error_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'out.json'
            predict_seams.write_error_payload(path, 'ExampleError', 'failed')

            payload = json.loads(path.read_text(encoding='utf-8'))

        self.assertEqual(payload['schema_version'], 1)
        self.assertEqual(payload['status'], 'error')
        self.assertEqual(payload['error_type'], 'ExampleError')
        self.assertEqual(payload['message'], 'failed')

    def test_device_resolution(self):
        with patch.object(predict_seams.torch.cuda, 'is_available', return_value=False):
            self.assertEqual(str(predict_seams.resolve_device('auto')), 'cpu')
            with self.assertRaisesRegex(predict_seams.PredictionError, 'CUDA is unavailable'):
                predict_seams.resolve_device('cuda')

    def test_prediction_model_kwargs_accept_legacy_gnn_hidden_key(self):
        kwargs = predict_seams.resolve_model_kwargs('gatv2', {
            'in_dim': 14,
            'hidden': 8,
            'num_layers': 1,
            'dropout': 0.0,
            'heads': 1,
        })

        self.assertEqual(kwargs['hidden_dim'], 8)

    def test_prediction_model_kwargs_accept_legacy_graphsage_mean_aggregation(self):
        kwargs = predict_seams.resolve_model_kwargs('graphsage', {
            'in_dim': 14,
            'hidden_dim': 8,
            'num_layers': 1,
            'dropout': 0.0,
            'skip_connections': 'hidden',
            'aggr': 'mean',
        })

        self.assertEqual(kwargs['aggr'], 'mean')

    def test_prediction_model_kwargs_require_sparsemeshcnn_model_config(self):
        kwargs = predict_seams.resolve_model_kwargs('sparsemeshcnn', {
            'model_config': {
                'in_channels': 14,
                'hidden_channels': 16,
                'dropout': 0.2,
                'pool_ratios': [0.85, 0.75],
                'min_edges': 32,
            },
            'feature_metadata': {'feature_dim': 14},
        })
        self.assertEqual(kwargs['in_channels'], 14)
        self.assertEqual(kwargs['hidden_channels'], 16)

        with self.assertRaisesRegex(predict_seams.PredictionError, 'model_config.*JSON object'):
            predict_seams.resolve_model_kwargs('sparsemeshcnn', {
                'in_channels': 14,
                'hidden_channels': 16,
                'dropout': 0.2,
            })

        with self.assertRaisesRegex(predict_seams.PredictionError, 'model_config.*JSON object'):
            predict_seams.resolve_model_kwargs('sparsemeshcnn', {
                'model_config': "{'in_channels': 14, 'hidden_channels': 16}",
            })

        with self.assertRaisesRegex(predict_seams.PredictionError, 'feature_metadata.*JSON object'):
            predict_seams.resolve_model_kwargs('sparsemeshcnn', {
                'model_config': {
                    'in_channels': 14,
                    'hidden_channels': 16,
                    'dropout': 0.2,
                    'pool_ratios': [0.85, 0.75],
                    'min_edges': 32,
                },
                'feature_metadata': "{'feature_dim': 14}",
            })

    def test_extract_state_dict_accepts_legacy_wrapper_keys(self):
        state = {'layer.weight': torch.zeros(1)}
        self.assertIs(predict_seams.extract_state_dict(state), state)
        self.assertEqual(predict_seams.extract_state_dict({'model_state': state}), state)
        for stale_key in ('state_dict', 'model_state_dict'):
            with self.subTest(stale_key=stale_key):
                self.assertEqual(predict_seams.extract_state_dict({stale_key: state}), state)

    def test_feature_metadata_rejects_stringified_feature_names(self):
        selection = predict_seams.resolve_feature_selection('paper14')
        config = {
            'feature_group': 'paper14',
            'feature_names': str(list(selection.feature_names)),
            'in_dim': selection.feature_count,
        }
        with self.assertRaisesRegex(predict_seams.PredictionError, 'feature_names'):
            predict_seams.validate_feature_metadata(config, {}, selection, {'in_dim': selection.feature_count})

    def test_feature_metadata_rejects_stringified_nested_metadata(self):
        selection = predict_seams.resolve_feature_selection('paper14')
        with self.assertRaisesRegex(predict_seams.PredictionError, 'feature_metadata.*JSON object'):
            predict_seams.validate_feature_metadata(
                {'feature_metadata': "{'feature_group': 'paper14'}"},
                {},
                selection,
                {'in_dim': selection.feature_count},
            )

    def test_auto_feature_inference_rejects_stringified_metadata(self):
        with self.assertRaisesRegex(predict_seams.PredictionError, 'feature_flags must be a JSON object'):
            predict_seams.infer_feature_bundle(
                {'feature_group': 'custom', 'feature_flags': "{'ao': True}"},
                {},
            )
        with self.assertRaisesRegex(predict_seams.PredictionError, 'feature_names must be a JSON list'):
            predict_seams.infer_feature_bundle(
                {'feature_names': "['pos_x_i']"},
                {},
            )

    def test_postprocess_kwargs_from_args(self):
        args = Namespace(
            postprocess_tau_low=0.25,
            postprocess_d_max=2,
            postprocess_r_bridge=5,
            postprocess_l_min=3,
            postprocess_anchor_boundary=False,
        )

        kwargs = predict_seams.postprocess_kwargs_from_args(args)

        self.assertEqual(kwargs, {
            'tau_low': 0.25,
            'd_max': 2,
            'r_bridge': 5,
            'l_min': 3,
            'anchor_boundary': False,
        })
        self.assertIs(type(kwargs['tau_low']), float)
        self.assertIs(type(kwargs['d_max']), int)
        self.assertIs(type(kwargs['anchor_boundary']), bool)

    def test_parser_rejects_internal_postprocess_stage_b_knobs(self):
        internal_flags = (
            '--postprocess-max-bridge-edges',
            '--postprocess-max-bridge-euclidean-ratio',
            '--postprocess-max-endpoint-candidates',
            '--postprocess-require-mutual-pairing',
            '--postprocess-min-loop-size-to-allow',
            '--postprocess-tangent-alignment-weight',
            '--postprocess-max-debug-candidates',
        )
        for flag in internal_flags:
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    predict_seams.parse_args([
                        '--mesh-path', 'mesh.obj',
                        '--model-weights', 'weights.pt',
                        '--output-json', 'out.json',
                        flag, '1',
                    ])

    def test_postprocess_path_invokes_pipeline_when_enabled(self):
        calls = {'pipeline': 0}
        real_pipeline = predict_seams.apply_topology_pipeline

        def fake_pipeline(**kwargs):
            calls['pipeline'] += 1
            return real_pipeline(
                view=kwargs['view'],
                probabilities=kwargs['probabilities'],
                topology=kwargs['topology'],
                tau_low=kwargs['tau_low'],
                d_max=kwargs['d_max'],
                r_bridge=kwargs['r_bridge'],
                l_min=kwargs['l_min'],
                anchor_boundary=kwargs['anchor_boundary'],
            )

        with self._patched_prediction_env(), \
                patch('predict_seams_bridge.apply_topology_pipeline', side_effect=fake_pipeline):
            with tempfile.TemporaryDirectory() as tmp:
                args = self._run_args(tmp)
                payload = predict_seams.run_prediction(args)

        telemetry = payload['diagnostics']['postprocess']
        self.assertEqual(calls['pipeline'], 1)
        json.dumps(telemetry)
        self.assertEqual(set(telemetry), {'parameters', 'skeleton', 'bridging', 'pruning', 'final_edge_count'})

    def test_postprocess_disabled_skips_pipeline(self):
        calls = {'pipeline': 0}

        def fake_pipeline(**kwargs):
            del kwargs
            calls['pipeline'] += 1
            raise AssertionError('pipeline should not run')

        with self._patched_prediction_env(probabilities=(0.2, 0.8, 0.6, 0.4, 0.9)), \
                patch('predict_seams_bridge.apply_topology_pipeline', side_effect=fake_pipeline):
            with tempfile.TemporaryDirectory() as tmp:
                args = self._run_args(tmp, postprocess=False)
                payload = predict_seams.run_prediction(args)

        self.assertEqual(calls['pipeline'], 0)
        self.assertEqual(payload['seam_edge_indices'], [1, 2, 4])
        self.assertNotIn('postprocess', payload.get('diagnostics', {}))

    def test_postprocess_failure_raises_clear_error(self):
        def fake_pipeline(**kwargs):
            del kwargs
            raise RuntimeError('deliberate pipeline failure')

        with self._patched_prediction_env(), \
                patch('predict_seams_bridge.apply_topology_pipeline', side_effect=fake_pipeline):
            with tempfile.TemporaryDirectory() as tmp:
                args = self._run_args(tmp)
                with self.assertRaises(predict_seams.PredictionError) as caught:
                    predict_seams.run_prediction(args)

        self.assertEqual(caught.exception.error_type, 'PostprocessFailed')
        self.assertIn('deliberate pipeline failure', str(caught.exception))

    def test_dual_edge_index_helper_matches_vertex_sharing_semantics(self):
        unique_edges = np.asarray([(0, 1), (0, 2), (1, 2), (2, 3)], dtype=np.int64)

        edge_index = build_dual_edge_index_from_unique_edges(unique_edges)

        self.assertEqual(edge_index.tolist(), [
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
            [1, 2, 0, 2, 3, 0, 1, 3, 1, 2],
        ])

    @contextlib.contextmanager
    def _patched_prediction_env(self, probabilities=(0.2, 0.8, 0.6, 0.4, 0.9)):
        topology = _square_topology()
        unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64)
        edge_features = np.zeros((len(unique_edges), 14), dtype=np.float32)
        with patch('predict_seams_bridge.compute_edge_features_for_selection', return_value=(edge_features, unique_edges, None)), \
                patch('predict_seams_bridge.build_prediction_model', return_value=_DummyModel(probabilities)), \
                patch('predict_seams_bridge.load_weights_payload', return_value={}):
            yield

    def _run_args(
        self,
        tmp: str,
        *,
        postprocess: bool = True,
    ) -> Namespace:
        tmp_path = Path(tmp)
        mesh_path = tmp_path / 'mesh.obj'
        weights_path = tmp_path / 'best_model.pth'
        config_path = tmp_path / 'config.json'
        summary_path = tmp_path / 'summary.json'
        output_path = tmp_path / 'out.json'
        mesh_path.write_text(SQUARE_OBJ, encoding='utf-8')
        weights_path.write_bytes(b'placeholder')
        config_path.write_text(json.dumps({
            'model': 'gatv2',
            'in_dim': 14,
            'hidden_dim': 8,
            'num_layers': 1,
            'dropout': 0.0,
            'heads': 1,
        }), encoding='utf-8')
        summary_path.write_text('{}', encoding='utf-8')
        args = predict_seams.parse_args([
            '--mesh-path', str(mesh_path),
            '--model-weights', str(weights_path),
            '--config-json', str(config_path),
            '--summary-json', str(summary_path),
            '--output-json', str(output_path),
            '--threshold', '0.5',
            '--device', 'cpu',
            '--model-type', 'gatv2',
            '--feature-bundle', 'paper14',
        ])
        args.postprocess = postprocess
        return args

    def _payload(self, write_all_edges):
        topology = _square_topology()
        unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64)
        probabilities = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3], dtype=np.float32)
        seam_mask = probabilities >= 0.75
        selection, _, _ = predict_seams.resolve_feature_bundle(_args('paper14'), {}, {})
        return predict_seams.build_output_payload(
            mesh_path=Path('mesh.obj'),
            output_json=Path('out.json'),
            weights_path=Path('best_model.pth'),
            config_path=Path('config.json'),
            summary_path=Path('summary.json'),
            model_type='gatv2',
            feature_bundle='paper14',
            selection=selection,
            threshold=0.75,
            device=torch.device('cpu'),
            topology=topology,
            unique_edges=unique_edges,
            probabilities=probabilities,
            seam_mask=seam_mask,
            write_all_edges=write_all_edges,
        )


class OutputPayloadModelTypeTests(unittest.TestCase):
    def _base_payload(self, model_type: str) -> dict:
        topology = _square_topology()
        unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64)
        probabilities = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3], dtype=np.float32)
        seam_mask = probabilities >= 0.75
        selection, _, _ = predict_seams.resolve_feature_bundle(_args('paper14'), {}, {})
        return predict_seams.build_output_payload(
            mesh_path=Path('mesh.obj'),
            output_json=Path('out.json'),
            weights_path=Path('best_model.pth'),
            config_path=Path('config.json'),
            summary_path=Path('summary.json'),
            model_type=model_type,
            feature_bundle='paper14',
            selection=selection,
            threshold=0.75,
            device=torch.device('cpu'),
            topology=topology,
            unique_edges=unique_edges,
            probabilities=probabilities,
            seam_mask=seam_mask,
            write_all_edges=False,
        )

    def test_sparsemeshcnn_model_type_stays_public_in_output(self):
        payload = self._base_payload('sparsemeshcnn')
        self.assertEqual(payload['model']['model_type'], 'sparsemeshcnn')
        self.assertNotIn('internal_model_type', payload['model'])

    def test_gatv2_model_type_unchanged_in_output(self):
        payload = self._base_payload('gatv2')
        self.assertEqual(payload['model']['model_type'], 'gatv2')
        self.assertNotIn('internal_model_type', payload['model'])

    def test_graphsage_model_type_unchanged_in_output(self):
        payload = self._base_payload('graphsage')
        self.assertEqual(payload['model']['model_type'], 'graphsage')
        self.assertNotIn('internal_model_type', payload['model'])


class ThicknessSdfFlagTests(unittest.TestCase):
    def test_parser_accepts_enable_thickness_sdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / 'mesh.obj'
            mesh.write_text('', encoding='utf-8')
            out = Path(tmp) / 'out.json'
            weights = Path(tmp) / 'weights.pth'
            weights.write_bytes(b'')
            args = predict_seams.parse_args([
                '--mesh-path', str(mesh),
                '--model-weights', str(weights),
                '--output-json', str(out),
                '--feature-bundle', 'custom',
                '--enable-ao',
                '--enable-thickness-sdf',
            ])
        self.assertTrue(args.enable_thickness_sdf)

    def test_parser_rejects_ao_density_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / 'mesh.obj'
            mesh.write_text('', encoding='utf-8')
            out = Path(tmp) / 'out.json'
            weights = Path(tmp) / 'weights.pth'
            weights.write_bytes(b'')
            with self.assertRaises(SystemExit):
                predict_seams.parse_args([
                    '--mesh-path', str(mesh),
                    '--model-weights', str(weights),
                    '--output-json', str(out),
                    '--feature-bundle', 'ao_density',
                ])

    def test_custom_bundle_with_sdf_flag_includes_sdf_feature(self):
        args = _args('custom', enable_ao=True, enable_thickness_sdf=True)
        selection, _, _ = predict_seams.resolve_feature_bundle(args, {}, {})
        self.assertTrue(any('sdf' in name or 'thickness' in name for name in selection.feature_names))

    def test_custom_bundle_without_sdf_flag_excludes_sdf_feature(self):
        args = _args('custom', enable_ao=True)
        selection, _, _ = predict_seams.resolve_feature_bundle(args, {}, {})
        self.assertFalse(any('sdf' in name or 'thickness' in name for name in selection.feature_names))

    def test_sdf_toggle_blocked_outside_custom_bundle(self):
        args = _args('paper14', enable_thickness_sdf=True)
        with self.assertRaises(predict_seams.PredictionError):
            predict_seams.resolve_feature_bundle(args, {}, {})

    def test_sdf_toggle_blocked_in_auto_bundle(self):
        args = _args('auto', enable_thickness_sdf=True)
        with self.assertRaises(predict_seams.PredictionError):
            predict_seams.resolve_feature_bundle(args, {}, {})

    def test_infer_feature_bundle_requires_metadata_in_auto_mode(self):
        with self.assertRaisesRegex(predict_seams.PredictionError, 'could not be inferred'):
            predict_seams.infer_feature_bundle({}, {})

    def test_infer_feature_bundle_rejects_custom_metadata_without_optional_flags(self):
        with self.assertRaisesRegex(predict_seams.PredictionError, 'does not specify any optional custom feature flags'):
            predict_seams.infer_feature_bundle(
                {'feature_group': 'custom', 'feature_flags': {}},
                {},
            )

    def test_infer_feature_bundle_rejects_legacy_paper_alias(self):
        with self.assertRaisesRegex(predict_seams.PredictionError, 'could not be inferred'):
            predict_seams.infer_feature_bundle(
                {'feature_group': 'paper'},
                {},
            )

    def test_validate_feature_metadata_rejects_legacy_custom_base_metadata(self):
        selection = predict_seams.resolve_feature_selection('paper14')
        config = {
            'feature_group': 'custom',
            'feature_flags': {},
            'feature_names': list(selection.feature_names),
            'in_dim': selection.feature_count,
        }
        with self.assertRaisesRegex(predict_seams.PredictionError, 'feature_group mismatch'):
            predict_seams.validate_feature_metadata(config, {}, selection, {'in_dim': selection.feature_count})

    def test_validate_feature_metadata_rejects_legacy_dihedral_flag_alias(self):
        selection = predict_seams.resolve_feature_selection('custom', enable_dihedral=True)
        config = {
            'feature_group': 'custom',
            'feature_flags': {'dihedral': True},
            'feature_names': list(selection.feature_names),
            'in_dim': selection.feature_count,
        }
        with self.assertRaisesRegex(predict_seams.PredictionError, 'unsupported key'):
            predict_seams.validate_feature_metadata(config, {}, selection, {'in_dim': selection.feature_count})


if __name__ == '__main__':
    unittest.main()
