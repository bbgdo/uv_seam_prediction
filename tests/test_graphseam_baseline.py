import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

from models.dual_graphsage.model import DualGraphSAGE
from models.dual_graphsage.train import validate_strict_paper_protocol
from models.baselines.registry import get_baseline
from tools.run_baseline import parse_args as parse_baseline_args
from models.utils.experiment_log import ExperimentLogger
from preprocessing.compute_features import compute_edge_features
from preprocessing.feature_registry import get_feature_group


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


class GraphSeamBaselineTests(unittest.TestCase):
    def test_baseline_registry_exposes_supported_models(self):
        self.assertIs(get_baseline('graphsage').model_class, DualGraphSAGE)
        self.assertEqual(get_baseline('gatv2').default_config_overrides['hidden_size'], 64)

    def test_unified_runner_defaults_graphsage_and_gatv2(self):
        graphsage_args = parse_baseline_args(['--epochs', '1'])
        gatv2_args = parse_baseline_args(['--model', 'gatv2', '--epochs', '1'])

        self.assertEqual(graphsage_args.model, 'graphsage')
        self.assertEqual(graphsage_args.hidden, 128)
        self.assertEqual(graphsage_args.lr, 1e-3)
        self.assertEqual(gatv2_args.model, 'gatv2')
        self.assertEqual(gatv2_args.hidden, 64)
        self.assertEqual(gatv2_args.lr, 5e-4)

    def test_feature_preset_shapes(self):
        mesh = _tiny_mesh()

        paper, edges, _ = compute_edge_features(mesh, feature_preset='paper14', endpoint_order='random')
        extended, extended_edges, _ = compute_edge_features(mesh, feature_preset='extended18')

        self.assertEqual(paper.shape, (len(edges), 14))
        self.assertEqual(extended.shape, (len(extended_edges), 18))

    def test_feature_registry_scaffold_lists_existing_baselines(self):
        self.assertEqual(get_feature_group('paper14').feature_preset, 'paper14')
        self.assertEqual(len(get_feature_group('extended18').feature_names), 18)

    def test_lstm_graphsage_forward(self):
        model = DualGraphSAGE(
            in_dim=14,
            hidden_dim=64,
            num_layers=3,
            aggr='lstm',
            skip_connections='all',
        )
        x = torch.randn(5, 14)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 0],
            [1, 2, 3, 4, 0, 2],
        ], dtype=torch.long)

        out = model(x, edge_index)

        self.assertEqual(out.shape, (5,))

    def test_strict_paper_protocol_accepts_valid_mocked_dataset(self):
        data = Data(x=torch.zeros(3, 14))
        data.label_source = 'exact_obj'
        data.feature_preset = 'paper14'
        args = Namespace(
            preset='paper',
            resolution_tag='all',
            in_dim=14,
            aggr='lstm',
            skip_connections='all',
        )

        validate_strict_paper_protocol(args, [data])

    def test_strict_paper_protocol_rejects_inconsistent_metadata(self):
        data = Data(x=torch.zeros(3, 14))
        data.label_source = 'legacy_uv_remap'
        data.feature_preset = 'extended18'
        args = Namespace(
            preset='paper',
            resolution_tag='all',
            in_dim=14,
            aggr='lstm',
            skip_connections='all',
        )

        with self.assertRaisesRegex(ValueError, 'strict paper protocol failed'):
            validate_strict_paper_protocol(args, [data])

    def test_strict_paper_protocol_requires_resolution_selector(self):
        data = Data(x=torch.zeros(3, 14))
        data.label_source = 'exact_obj'
        data.feature_preset = 'paper14'
        args = Namespace(
            preset='paper',
            resolution_tag=None,
            in_dim=14,
            aggr='lstm',
            skip_connections='all',
        )

        with self.assertRaisesRegex(ValueError, 'resolution_tag must be set'):
            validate_strict_paper_protocol(args, [data])

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
