import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

from models.gatv2.model import DualGATv2
from models.dual_graphsage.model import DualGraphSAGE
from models.common.gnn_train_data import apply_runtime_feature_selection
from models.common.gnn_train_runtime import build_runtime_config, logger_config, model_kwargs
from models.common.gnn_registry import get_gnn_model
from tools.run_training import parse_args as parse_training_args
from models.utils.experiment_log import ExperimentLogger
from preprocessing.compute_features import compute_edge_features
from preprocessing.feature_registry import get_feature_group, resolve_feature_selection
from preprocessing.build_gnn_dataset import build_dual_data


def _tiny_mesh() -> trimesh.Trimesh:
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    faces = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _skewed_density_mesh() -> trimesh.Trimesh:
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1e-9, 0.0, 0.0],
        [0.0, 1e-9, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [1, 3, 4],
        [1, 4, 2],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class GraphSeamBaselineTests(unittest.TestCase):
    def test_gnn_registry_exposes_supported_models(self):
        self.assertIs(get_gnn_model('graphsage').model_class, DualGraphSAGE)
        self.assertIs(get_gnn_model('gatv2').model_class, DualGATv2)
        self.assertEqual(get_gnn_model('graphsage').gnn_config_overrides, {})
        self.assertEqual(get_gnn_model('gatv2').gnn_config_overrides['hidden_size'], 64)
        self.assertEqual(get_gnn_model('gatv2').gnn_config_overrides['heads'], 4)

    def test_unified_runner_defaults_graphsage_and_gatv2(self):
        graphsage_args = parse_training_args(['--epochs', '1'])
        gatv2_args = parse_training_args(['--model', 'gatv2', '--epochs', '1'])

        self.assertEqual(graphsage_args.model, 'graphsage')
        self.assertEqual(graphsage_args.hidden, 128)
        self.assertEqual(graphsage_args.lr, 3e-4)
        self.assertEqual(gatv2_args.model, 'gatv2')
        self.assertEqual(gatv2_args.hidden, 64)
        self.assertEqual(gatv2_args.heads, 4)
        self.assertEqual(gatv2_args.lr, 3e-4)

    def test_feature_group_shapes(self):
        mesh = _tiny_mesh()

        paper, edges, _ = compute_edge_features(mesh, feature_group='paper14', endpoint_order='random')

        self.assertEqual(paper.shape, (len(edges), 14))

    def test_feature_registry_rejects_extended18(self):
        with self.assertRaises(ValueError):
            get_feature_group('extended18')
        with self.assertRaises(ValueError):
            resolve_feature_selection('extended18')

    def test_feature_registry_exposes_locked_paper14_bundle(self):
        self.assertEqual(get_feature_group('paper14').name, 'paper14')
        with self.assertRaisesRegex(ValueError, 'custom requires'):
            get_feature_group('custom')

    def test_feature_registry_resolves_custom_toggles(self):
        paper = resolve_feature_selection('paper14')
        ao_only = resolve_feature_selection('custom', enable_ao=True)
        symmetry_only = resolve_feature_selection('custom', enable_symmetry=True)
        density_only = resolve_feature_selection('custom', enable_density=True)
        combined = resolve_feature_selection(
            'custom',
            enable_ao=True,
            enable_symmetry=True,
            enable_density=True,
        )

        self.assertEqual(paper.feature_count, 14)
        self.assertEqual(ao_only.feature_names[-1], 'ao_j')
        self.assertEqual(symmetry_only.feature_names[-1], 'symmetry_dist')
        self.assertEqual(density_only.feature_names[-2:], ('density_mean', 'density_diff'))
        self.assertEqual(combined.feature_count, 19)

    def test_feature_registry_rejects_custom_without_optional_features(self):
        with self.assertRaisesRegex(ValueError, "custom.*requires at least one optional feature"):
            resolve_feature_selection('custom')

    def test_feature_registry_rejects_toggles_on_locked_bundle(self):
        with self.assertRaisesRegex(ValueError, 'require feature_group=.custom.'):
            resolve_feature_selection('paper14', enable_density=True)

    def test_density_features_are_finite_on_tiny_mesh(self):
        mesh = _tiny_mesh()

        features, edges, _ = compute_edge_features(
            mesh,
            feature_group='custom',
            enable_density=True,
        )

        self.assertEqual(features.shape, (len(edges), 16))
        self.assertTrue(np.isfinite(features[:, -2:]).all())

    def test_density_features_are_bounded_after_normalization(self):
        mesh = _skewed_density_mesh()

        features, _, _ = compute_edge_features(
            mesh,
            feature_group='custom',
            enable_density=True,
        )
        density_mean = features[:, -2]
        density_diff = features[:, -1]
        tol = 1e-6

        self.assertTrue(np.isfinite(features[:, -2:]).all())
        self.assertTrue(np.all(density_mean >= -1.0 - tol))
        self.assertTrue(np.all(density_mean <= 1.0 + tol))
        self.assertTrue(np.all(density_diff >= 0.0 - tol))
        self.assertTrue(np.all(density_diff <= 1.0 + tol))

    def test_dual_graph_preserves_feature_metadata(self):
        data = Data(
            edge_index=torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long),
            edge_attr=torch.zeros(4, 16),
            y=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            num_nodes=2,
        )
        data.feature_names = list(resolve_feature_selection('custom', enable_density=True).feature_names)
        data.feature_group = 'custom'
        data.feature_flags = {'ao': False, 'signed_dihedral': False, 'symmetry': False, 'density': True}
        data.density_config = {'neighborhood': '2-ring'}

        dual = build_dual_data(data)

        self.assertEqual(dual.feature_names, data.feature_names)
        self.assertEqual(dual.feature_group, 'custom')
        self.assertEqual(dual.feature_flags['density'], True)
        self.assertEqual(dual.density_config['neighborhood'], '2-ring')

    def test_runtime_feature_slicing_uses_metadata_superset(self):
        superset = resolve_feature_selection(
            'custom',
            enable_ao=True,
            enable_symmetry=True,
            enable_density=True,
        )
        requested = resolve_feature_selection('custom', enable_symmetry=True)
        data = Data(x=torch.arange(2 * superset.feature_count, dtype=torch.float32).reshape(2, -1))
        data.feature_names = list(superset.feature_names)

        apply_runtime_feature_selection([data], requested)

        expected_idx = [superset.feature_names.index(name) for name in requested.feature_names]
        expected = torch.arange(2 * superset.feature_count, dtype=torch.float32).reshape(2, -1)[:, expected_idx]
        self.assertTrue(torch.equal(data.x, expected))
        self.assertEqual(data.feature_names, list(requested.feature_names))

    def test_runtime_feature_slicing_requires_metadata_for_missing_columns(self):
        requested = resolve_feature_selection('custom', enable_density=True)
        data = Data(x=torch.zeros(2, 18))

        with self.assertRaisesRegex(ValueError, 'missing feature_names metadata'):
            apply_runtime_feature_selection([data], requested)

    def test_runtime_feature_selection_requires_paper14_feature_names(self):
        requested = resolve_feature_selection('paper14')
        data = Data(x=torch.zeros(2, 14))
        data.feature_group = 'paper14'

        with self.assertRaisesRegex(ValueError, 'missing feature_names metadata'):
            apply_runtime_feature_selection([data], requested)

    def test_gatv2_forward_returns_one_logit_per_dual_node(self):
        model = DualGATv2(in_dim=14, hidden_dim=32, heads=4, num_layers=3, dropout=0.1)
        x = torch.randn(5, 14)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 0],
            [1, 2, 3, 4, 0, 2],
        ], dtype=torch.long)

        out = model(x, edge_index)

        self.assertEqual(out.shape, (5,))

    def test_shared_trainer_instantiates_gatv2_with_runtime_dims(self):
        args = parse_training_args([
            '--model', 'gatv2',
            '--epochs', '1',
            '--feature-group', 'custom',
            '--enable-ao',
            '--enable-density',
        ])
        selection = resolve_feature_selection('custom', enable_ao=True, enable_density=True)
        args.in_dim = selection.feature_count
        config = build_runtime_config(args)
        definition = get_gnn_model(config.model_name)

        model = definition.model_class(**model_kwargs(config))
        x = torch.randn(4, selection.feature_count)
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ], dtype=torch.long)

        self.assertEqual(model(x, edge_index).shape, (4,))

    def test_gnn_logger_config_uses_canonical_metadata_fields(self):
        args = parse_training_args([
            '--model', 'gatv2',
            '--feature-group', 'custom',
            '--enable-ao',
            '--resolution-tag', 'all',
        ])
        config = build_runtime_config(args)
        payload = logger_config(
            args,
            config,
            'GATv2',
            torch.tensor([1.0]),
            {'train': [], 'val': [], 'test': []},
            {'graph_count': 0},
            0,
            33,
            (0, 0, 0),
        )

        self.assertEqual(payload['hidden_dim'], config.hidden_size)
        self.assertEqual(payload['resolution_tag'], 'all')
        self.assertNotIn('hidden', payload)
        self.assertNotIn('resolution_selector', payload)

    def test_lstm_graphsage_forward(self):
        model = DualGraphSAGE(
            in_dim=14,
            hidden_dim=64,
            num_layers=3,
            skip_connections='all',
        )
        x = torch.randn(5, 14)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 0],
            [1, 2, 3, 4, 0, 2],
        ], dtype=torch.long)

        out = model(x, edge_index)

        self.assertEqual(out.shape, (5,))

    def test_summary_export_contains_threshold_metrics(self):
        metrics_05 = {
            'f1': 0.4,
            'precision': 0.5,
            'recall': 0.3333,
            'accuracy': 0.7,
            'fpr': 0.1,
            'tpr': 0.3333,
            'tp': 1,
            'fp': 1,
            'fn': 2,
            'tn': 6,
        }
        metrics_best = {
            'f1': 0.6,
            'precision': 0.75,
            'recall': 0.5,
            'accuracy': 0.8,
            'fpr': 0.05,
            'tpr': 0.5,
            'tp': 2,
            'fp': 1,
            'fn': 2,
            'tn': 7,
        }

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            logger = ExperimentLogger(run_dir=run_dir, config={'model': 'DualGraphSAGE'})
            logger.log_epoch(epoch=1, val_f1=0.5, val_precision=0.5, val_recall=0.5)
            logger.finalize(
                test_metrics=metrics_05,
                best_epoch=1,
                extra_summary={
                    'best_validation_threshold': 0.7,
                    'test_metrics_threshold_0_5': metrics_05,
                    'test_metrics_best_validation_threshold': metrics_best,
                    'test_confusion_threshold_0_5': {'tp': 1, 'fp': 1, 'fn': 2, 'tn': 6},
                    'test_confusion_best_validation_threshold': {'tp': 2, 'fp': 1, 'fn': 2, 'tn': 7},
                },
            )
            logger.save()

            with open(run_dir / 'summary.json') as f:
                summary = json.load(f)

        self.assertEqual(summary['best_validation_threshold'], 0.7)
        self.assertEqual(summary['test_metrics_threshold_0_5']['f1'], 0.4)
        self.assertEqual(summary['test_metrics_best_validation_threshold']['f1'], 0.6)
        self.assertEqual(summary['test_confusion_threshold_0_5']['tp'], 1)
        self.assertEqual(summary['test_confusion_best_validation_threshold']['tp'], 2)


if __name__ == '__main__':
    unittest.main()
