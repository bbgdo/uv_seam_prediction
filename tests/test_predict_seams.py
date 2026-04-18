import importlib.util
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

from preprocessing.build_dual_graph import build_dual_edge_index_from_unique_edges
from preprocessing.obj_parser import parse_obj_text
from preprocessing.topology import WeldConfig, build_topology


SQUARE_OBJ = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3
f 1 3 4
"""


def _args(feature_bundle='paper14_locked', **overrides):
    values = {
        'feature_bundle': feature_bundle,
        'enable_ao': False,
        'enable_dihedral': False,
        'enable_symmetry': False,
        'enable_density': False,
    }
    values.update(overrides)
    return Namespace(**values)


def _square_topology():
    return build_topology(parse_obj_text(SQUARE_OBJ), WeldConfig.exact())


class PredictSeamsTests(unittest.TestCase):
    def test_threshold_resolution_precedence(self):
        summary = {'best_validation_threshold': 0.7}

        self.assertEqual(predict_seams.resolve_threshold(0.8, summary), 0.8)
        self.assertEqual(predict_seams.resolve_threshold(None, summary), 0.7)
        with self.assertRaisesRegex(predict_seams.PredictionError, 'threshold is required'):
            predict_seams.resolve_threshold(None, {})

    def test_model_type_resolution_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / 'gatv2_run' / 'best_model.pth'

            self.assertEqual(predict_seams.resolve_model_type('graphsage', {'model': 'gatv2'}, weights), 'graphsage')
            self.assertEqual(predict_seams.resolve_model_type('auto', {'model_name': 'DualGraphSAGE'}, weights), 'graphsage')
            self.assertEqual(predict_seams.resolve_model_type('auto', {}, weights), 'gatv2')
            self.assertEqual(predict_seams.resolve_model_type('auto', {'model': 'meshcnn_full'}, weights), 'meshcnn_full')
            self.assertEqual(predict_seams.resolve_model_type('sparsemeshcnn', {}, weights), 'meshcnn_full')

            with self.assertRaisesRegex(predict_seams.PredictionError, 'model type could not be resolved'):
                predict_seams.resolve_model_type('auto', {}, Path(tmp) / 'run' / 'best_model.pth')

    def test_feature_bundle_resolution(self):
        selection, endpoint_order, _ = predict_seams.resolve_feature_bundle(_args('paper14_locked'), {}, {})
        self.assertEqual(selection.feature_group, 'paper14')
        self.assertEqual(selection.feature_count, 14)
        self.assertEqual(endpoint_order, 'random')

        selection, endpoint_order, _ = predict_seams.resolve_feature_bundle(_args('ao_density'), {}, {})
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
        selection, _, _ = predict_seams.resolve_feature_bundle(_args('paper14_locked'), {}, {})

        payload = predict_seams.build_output_payload(
            mesh_path=Path('mesh.obj'),
            output_json=Path('out.json'),
            weights_path=Path('best_model.pth'),
            config_path=Path('config.json'),
            summary_path=Path('summary.json'),
            model_type='gatv2',
            feature_bundle='paper14_locked',
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
        selection, endpoint_order, _ = predict_seams.resolve_feature_bundle(_args('paper14_locked'), {}, {})
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

    def test_dual_edge_index_helper_matches_vertex_sharing_semantics(self):
        unique_edges = np.asarray([(0, 1), (0, 2), (1, 2), (2, 3)], dtype=np.int64)

        edge_index = build_dual_edge_index_from_unique_edges(unique_edges)

        self.assertEqual(edge_index.tolist(), [
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
            [1, 2, 0, 2, 3, 0, 1, 3, 1, 2],
        ])

    def _payload(self, write_all_edges):
        topology = _square_topology()
        unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64)
        probabilities = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3], dtype=np.float32)
        seam_mask = probabilities >= 0.75
        selection, _, _ = predict_seams.resolve_feature_bundle(_args('paper14_locked'), {}, {})
        return predict_seams.build_output_payload(
            mesh_path=Path('mesh.obj'),
            output_json=Path('out.json'),
            weights_path=Path('best_model.pth'),
            config_path=Path('config.json'),
            summary_path=Path('summary.json'),
            model_type='gatv2',
            feature_bundle='paper14_locked',
            selection=selection,
            threshold=0.75,
            device=torch.device('cpu'),
            topology=topology,
            unique_edges=unique_edges,
            probabilities=probabilities,
            seam_mask=seam_mask,
            write_all_edges=write_all_edges,
        )


if __name__ == '__main__':
    unittest.main()
