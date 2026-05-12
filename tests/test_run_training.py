import unittest
from argparse import Namespace
from unittest.mock import patch

from models.common.gnn_train_runtime import build_runtime_config
from tools import run_training


class RunTrainingTests(unittest.TestCase):
    def test_parse_graphsage_defaults(self):
        args = run_training.parse_args(['--model', 'graphsage', '--epochs', '1'])

        self.assertEqual(args.model, 'graphsage')
        self.assertEqual(args.dataset, 'dataset_dual.pt')
        self.assertEqual(args.feature_group, None)
        self.assertFalse(args.mean_debug)
        self.assertTrue(args.run_dir.startswith('runs/dual_graphsage_'))

    def test_mean_debug_switches_only_graphsage_to_mean_aggregation(self):
        graphsage_args = run_training.parse_args(['--model', 'graphsage', '--epochs', '1', '--mean_debug'])
        gatv2_args = run_training.parse_args(['--model', 'gatv2', '--epochs', '1', '--mean_debug'])

        self.assertEqual(build_runtime_config(graphsage_args).aggr, 'mean')
        self.assertEqual(build_runtime_config(gatv2_args).aggr, 'lstm')

    def test_parse_gatv2_defaults(self):
        args = run_training.parse_args(['--model', 'gatv2', '--epochs', '1'])

        self.assertEqual(args.model, 'gatv2')
        self.assertEqual(args.dataset, 'dataset_dual.pt')
        self.assertEqual(args.heads, 4)
        self.assertTrue(args.run_dir.startswith('runs/gatv2_'))

    def test_parse_sparsemeshcnn_defaults(self):
        args = run_training.parse_args(['--model', 'sparsemeshcnn', '--epochs', '1'])

        self.assertEqual(args.model, 'sparsemeshcnn')
        self.assertEqual(args.dataset, 'dataset_sparsemeshcnn_paper14.pt')
        self.assertEqual(args.feature_group, 'paper14')
        self.assertEqual(args.lr, 3e-4)
        self.assertEqual(args.pool_ratios, '0.85,0.75')
        self.assertEqual(args.min_edges, 32)
        self.assertTrue(args.run_dir.startswith('runs/sparsemeshcnn_'))

    def test_dispatches_graph_model_to_gnn_trainer(self):
        args = run_training.parse_args(['--model', 'graphsage', '--epochs', '1'])

        with patch.object(run_training, 'train_gnn') as train_gnn:
            run_training.train_model(args)

        train_gnn.assert_called_once_with(args)

    def test_dispatches_sparsemeshcnn_to_sparse_trainer(self):
        args = run_training.parse_args(['--model', 'sparsemeshcnn', '--epochs', '1'])

        with patch.object(run_training, 'train_sparsemeshcnn') as train_sparsemeshcnn:
            run_training.train_model(args)

        train_sparsemeshcnn.assert_called_once_with(args)

    def test_architecture_script_accepts_namespace(self):
        from models.dual_graphsage import train as graphsage_train

        args = Namespace(dataset='dataset.pt', epochs=1)
        with patch.object(run_training, 'train_gnn') as train_gnn:
            graphsage_train.main(args)

        train_gnn.assert_called_once()
        self.assertEqual(train_gnn.call_args.args[0].model, 'graphsage')

    def test_sparsemeshcnn_script_accepts_namespace(self):
        from models.meshcnn_full import train as sparsemeshcnn_train

        args = Namespace(dataset='dataset.pt', epochs=1)
        with patch.object(run_training, 'train_sparsemeshcnn') as train_sparsemeshcnn:
            sparsemeshcnn_train.main(args)

        train_sparsemeshcnn.assert_called_once()
        self.assertEqual(train_sparsemeshcnn.call_args.args[0].model, 'sparsemeshcnn')


if __name__ == '__main__':
    unittest.main()
