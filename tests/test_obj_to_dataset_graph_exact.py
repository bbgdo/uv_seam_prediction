from contextlib import contextmanager
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from preprocessing.compute_features import compute_edge_features
from preprocessing.obj_parser import parse_obj
from preprocessing.obj_to_dataset_graph import (
    _build_feature_mesh_from_topology,
    build_dual_data,
    main as build_dataset_main,
    manifest_path_for_dataset,
    process_mesh,
)
from preprocessing.seam_labels import extract_seam_truth
from preprocessing.topology import WeldConfig, build_topology


NON_SEAM_SHARED_EDGE = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""


@contextmanager
def _obj_file(text: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / 'fixture.obj'
        path.write_text(text, encoding='utf-8')
        yield path


@contextmanager
def _mesh_dir(text: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_dir = Path(temp_dir) / 'meshes'
        mesh_dir.mkdir()
        (mesh_dir / 'fixture.obj').write_text(text, encoding='utf-8')
        yield mesh_dir


def _topology_and_truth(path: Path):
    topology = build_topology(parse_obj(path), WeldConfig.exact())
    return topology, extract_seam_truth(topology)


class ExactObjDatasetGraphTests(unittest.TestCase):
    def test_exact_obj_labels_follow_unique_edge_order(self):
        with _obj_file(NON_SEAM_SHARED_EDGE) as path:
            topology, seam_truth = _topology_and_truth(path)

            data = process_mesh(
                path,
                feature_preset='paper14',
                endpoint_order='fixed',
            )

            expected_edges = [list(edge) for edge in topology.canonical_edges]
            expected_labels = [1.0 if seam_truth.seam_map[edge] else 0.0 for edge in topology.canonical_edges]
            edge_count = len(expected_edges)

            self.assertEqual(data.unique_edges.tolist(), expected_edges)
            self.assertEqual(data.y[:edge_count].tolist(), expected_labels)
            self.assertEqual(data.y[edge_count:].tolist(), expected_labels)

    def test_exact_obj_boundary_edges_match_seam_truth(self):
        with _obj_file(NON_SEAM_SHARED_EDGE) as path:
            topology, seam_truth = _topology_and_truth(path)

            data = process_mesh(
                path,
                feature_preset='paper14',
                endpoint_order='fixed',
            )

            labels_by_edge = {
                tuple(edge): bool(label)
                for edge, label in zip(data.unique_edges.tolist(), data.y[:len(data.unique_edges)].tolist())
            }
            boundary_edges = {edge for edge, is_boundary in seam_truth.boundary_map.items() if is_boundary}

            self.assertEqual(data.boundary_edge_count, seam_truth.audit.boundary_edges)
            self.assertEqual(boundary_edges, {(0, 1), (0, 3), (1, 2), (2, 3)})
            self.assertFalse(seam_truth.boundary_map[(0, 2)])
            self.assertFalse(labels_by_edge[(0, 2)])
            self.assertTrue(all(labels_by_edge[edge] for edge in boundary_edges))

    def test_exact_obj_data_contains_audit_metadata(self):
        with _obj_file(NON_SEAM_SHARED_EDGE) as path:
            data = process_mesh(
                path,
                feature_preset='paper14',
                endpoint_order='fixed',
            )
            edge_count = len(data.unique_edges)

            self.assertEqual(data.file_path, str(path))
            self.assertEqual(data.label_source, 'exact_obj')
            self.assertEqual(data.feature_preset, 'paper14')
            self.assertEqual(data.endpoint_order, 'fixed')
            self.assertEqual(data.weld_mode, 'exact')
            self.assertEqual(data.unique_edges.shape, (edge_count, 2))
            self.assertEqual(data.seam_edge_count, int(data.y[:edge_count].sum().item()))
            self.assertEqual(data.boundary_edge_count, 4)

    def test_exact_obj_is_default_label_source(self):
        with _obj_file(NON_SEAM_SHARED_EDGE) as path:
            data = process_mesh(
                path,
                feature_preset='paper14',
                endpoint_order='fixed',
            )

            self.assertEqual(data.label_source, 'exact_obj')

    def test_feature_mesh_from_canonical_topology_preserves_edge_order(self):
        with _obj_file(NON_SEAM_SHARED_EDGE) as path:
            topology, _ = _topology_and_truth(path)
            feature_mesh = _build_feature_mesh_from_topology(topology)

            _, unique_edges, _ = compute_edge_features(
                feature_mesh,
                feature_preset='paper14',
                endpoint_order='fixed',
            )

            expected_edges = np.asarray(topology.canonical_edges, dtype=np.int64)
            self.assertTrue(np.array_equal(unique_edges, expected_edges))

    def test_saving_exact_obj_dataset_writes_manifest(self):
        with _mesh_dir(NON_SEAM_SHARED_EDGE) as mesh_dir:
            output_path = mesh_dir.parent / 'dataset_v2_exact_labels.pt'

            build_dataset_main([
                str(mesh_dir),
                '--max-meshes', '1',
                '--feature-preset', 'paper14',
                '--endpoint-order', 'fixed',
                '--save',
                '--output', str(output_path),
            ])

            manifest_path = manifest_path_for_dataset(output_path)
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())

            dataset = torch.load(output_path, weights_only=False)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].label_source, 'exact_obj')

    def test_exact_obj_manifest_has_required_fields(self):
        required_top_level = {
            'dataset_path',
            'label_source',
            'feature_preset',
            'endpoint_order',
            'weld_mode',
            'mesh_count',
            'total_nodes',
            'total_unique_edges',
            'total_directed_edges',
            'total_seam_edges',
            'total_boundary_edges',
            'aggregate_seam_ratio',
            'aggregate_pos_weight',
            'meshes',
        }
        required_mesh_fields = {
            'file_path',
            'nodes',
            'unique_edges',
            'seam_edges',
            'boundary_edges',
            'feature_dim',
        }

        with _mesh_dir(NON_SEAM_SHARED_EDGE) as mesh_dir:
            output_path = mesh_dir.parent / 'dataset_v2_exact_labels.pt'
            build_dataset_main([
                str(mesh_dir),
                '--max-meshes', '1',
                '--feature-preset', 'paper14',
                '--endpoint-order', 'fixed',
                '--save',
                '--output', str(output_path),
            ])

            manifest = json.loads(manifest_path_for_dataset(output_path).read_text(encoding='utf-8'))

            self.assertTrue(required_top_level.issubset(manifest.keys()))
            self.assertTrue(required_mesh_fields.issubset(manifest['meshes'][0].keys()))
            self.assertEqual(manifest['label_source'], 'exact_obj')
            self.assertEqual(manifest['feature_preset'], 'paper14')
            self.assertEqual(manifest['endpoint_order'], 'fixed')
            self.assertEqual(manifest['weld_mode'], 'exact')
            self.assertEqual(manifest['mesh_count'], 1)
            self.assertEqual(manifest['total_unique_edges'], 5)
            self.assertEqual(manifest['total_directed_edges'], 10)
            self.assertEqual(manifest['total_seam_edges'], 4)
            self.assertEqual(manifest['total_boundary_edges'], 4)
            self.assertAlmostEqual(manifest['aggregate_seam_ratio'], 4 / 5)
            self.assertAlmostEqual(manifest['aggregate_pos_weight'], 1 / 4)

    def test_dual_graph_preserves_exact_obj_metadata(self):
        with _obj_file(NON_SEAM_SHARED_EDGE) as path:
            data = process_mesh(
                path,
                feature_preset='paper14',
                endpoint_order='fixed',
            )

            dual = build_dual_data(data)

            self.assertEqual(dual.file_path, str(path))
            self.assertEqual(dual.label_source, 'exact_obj')
            self.assertEqual(dual.feature_preset, 'paper14')
            self.assertEqual(dual.endpoint_order, 'fixed')
            self.assertEqual(dual.weld_mode, 'exact')
            self.assertEqual(dual.seam_edge_count, data.seam_edge_count)
            self.assertEqual(dual.boundary_edge_count, data.boundary_edge_count)

if __name__ == '__main__':
    unittest.main()
