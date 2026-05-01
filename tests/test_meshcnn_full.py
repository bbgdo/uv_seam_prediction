import unittest
from contextlib import contextmanager
from pathlib import Path
import tempfile

import numpy as np
import torch

from models.meshcnn_full.mesh import MeshCNNSample, MutableMeshTopology, build_mesh_adjacency
from models.meshcnn_full.model import MeshCNNSegmenter
from models.meshcnn_full.pool import MeshPool
from models.meshcnn_full.train import slice_meshcnn_dataset_features
from models.meshcnn_full.unpool import MeshUnpool
from preprocessing.build_meshcnn_dataset import build_dataset_manifest, build_meshcnn_sample
from preprocessing.feature_registry import PAPER14_FEATURE_NAMES, resolve_feature_selection


OBJ_TWO_TRIANGLES = """
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

OBJ_TETRA = """
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
vt 0 0
vt 1 0
vt 0 1
vt 1 1
f 1/1 2/2 3/3
f 1/1 4/4 2/2
f 2/2 4/4 3/3
f 1/1 3/3 4/4
"""


@contextmanager
def _obj_file(text: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / 'mesh.obj'
        path.write_text(text, encoding='utf-8')
        yield path


def _valid_collapse_mesh():
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 1.0, 0.2],
        [1.0, 1.0, 0.5],
        [1.0, 0.0, 1.0],
        [-0.5, 0.0, 1.0],
    ], dtype=np.float32)
    faces = np.asarray([
        [0, 1, 2],
        [1, 0, 3],
        [0, 2, 4],
        [2, 1, 5],
        [1, 3, 6],
        [3, 0, 7],
    ], dtype=np.int64)
    return vertices, faces


def _nonmanifold_result_mesh():
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 1.0, 0.2],
        [0.5, 1.5, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float32)
    faces = np.asarray([
        [0, 1, 2],
        [1, 0, 3],
        [0, 4, 5],
        [4, 0, 6],
        [1, 4, 7],
    ], dtype=np.int64)
    return vertices, faces


def _full_custom_selection():
    return resolve_feature_selection(
        'custom',
        enable_ao=True,
        enable_dihedral=True,
        enable_symmetry=True,
        enable_density=True,
        enable_thickness_sdf=True,
    )


def _sample_with_features(feature_names: list[str] | tuple[str, ...] | None = None) -> MeshCNNSample:
    names = list(feature_names or _full_custom_selection().feature_names)
    edge_count = 4
    feature_dim = len(names)
    faces = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
    return MeshCNNSample(
        vertices=torch.zeros(4, 3),
        faces=faces,
        unique_edges=torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2]], dtype=torch.long),
        edge_features=torch.arange(edge_count * feature_dim, dtype=torch.float32).reshape(edge_count, feature_dim),
        edge_labels=torch.zeros(edge_count),
        edge_neighbors=torch.full((edge_count, 4), -1, dtype=torch.long),
        edge_to_faces=torch.full((edge_count, 2), -1, dtype=torch.long),
        face_to_edges=torch.zeros(2, 3, dtype=torch.long),
        boundary_mask=torch.ones(edge_count, dtype=torch.bool),
        file_path='toy.obj',
        feature_group='custom',
        feature_preset='custom',
        feature_names=names,
        feature_flags=_full_custom_selection().feature_flags.as_dict(),
        endpoint_order='random',
        label_source='exact_obj',
        density_config=_full_custom_selection().density_config,
    )


class MeshCNNFullTests(unittest.TestCase):
    def test_topology_reconstruction_matches_cached_arrays(self):
        with _obj_file(OBJ_TWO_TRIANGLES) as path:
            sample = build_meshcnn_sample(
                path,
                resolve_feature_selection('paper14'),
                endpoint_order='fixed',
            )
            rebuilt = build_mesh_adjacency(
                sample.faces.numpy(),
                sample.unique_edges.numpy(),
            )
            self.assertTrue(np.array_equal(rebuilt[0], sample.unique_edges.numpy()))
            self.assertTrue(np.array_equal(rebuilt[1], sample.edge_to_faces.numpy()))
            self.assertTrue(np.array_equal(rebuilt[2], sample.face_to_edges.numpy()))
            self.assertTrue(np.array_equal(rebuilt[3], sample.edge_neighbors.numpy()))
            self.assertTrue(np.array_equal(rebuilt[4], sample.boundary_mask.numpy()))

    def test_invalid_edge_collapses_are_rejected(self):
        vertices, faces = _valid_collapse_mesh()
        topology = MutableMeshTopology(vertices, faces)
        boundary_idx = int(np.flatnonzero(topology.boundary_mask)[0])
        self.assertEqual(topology.collapse_error(boundary_idx), 'boundary edge')

        vertices, faces = _nonmanifold_result_mesh()
        topology = MutableMeshTopology(vertices, faces)
        collapse_idx = topology.edge_key_to_idx[(0, 1)]
        self.assertEqual(topology.collapse_error(collapse_idx), 'non-manifold result')

    def test_valid_collapse_history_unpools_to_original_shape(self):
        vertices, faces = _valid_collapse_mesh()
        topology = MutableMeshTopology(vertices, faces)
        x = torch.randn(topology.edge_count, 8, requires_grad=True)
        pool = MeshPool(channels=8, target_ratio=0.5, min_edges=1, max_collapses=1)
        pooled, _, history = pool(x, topology)
        restored = MeshUnpool()(pooled, history)
        pooled.sum().backward()

        self.assertLessEqual(pooled.shape[0], x.shape[0])
        self.assertEqual(restored.shape, x.shape)
        self.assertEqual(history.old_edge_count, x.shape[0])
        self.assertEqual(history.new_edge_count, pooled.shape[0])
        self.assertTrue(any(param.grad is not None for param in pool.scorer.parameters()))

    def test_pool_exhausts_invalid_candidates_without_spinning(self):
        vertices = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        topology = MutableMeshTopology(vertices, faces)
        x = torch.randn(topology.edge_count, 4)
        pool = MeshPool(channels=4, target_ratio=0.1, min_edges=1, max_collapses=None)

        pooled, pooled_topology, history = pool(x, topology)
        debug = pool.get_last_debug()

        self.assertEqual(pooled.shape[0], topology.edge_count)
        self.assertEqual(pooled_topology.edge_count, topology.edge_count)
        self.assertEqual(history.new_edge_count, topology.edge_count)
        self.assertEqual(debug['attempted_collapses'], topology.edge_count)
        self.assertEqual(debug['successful_collapses'], 0)
        self.assertEqual(debug['rejected_collapses'], topology.edge_count)
        self.assertEqual(debug['stop_reason'], 'stagnated_no_valid_collapses')

    def test_forward_pass_on_cached_meshcnn_sample(self):
        with _obj_file(OBJ_TETRA) as path:
            sample = build_meshcnn_sample(
                path,
                resolve_feature_selection('paper14'),
                endpoint_order='fixed',
            )
            model = MeshCNNSegmenter(
                in_channels=sample.in_channels,
                hidden_channels=16,
                pool_ratios=(0.9, 0.9),
                min_edges=1,
                max_pool_collapses=4,
            )
            logits = model(sample)
            self.assertEqual(logits.shape, sample.edge_labels.shape)
            self.assertTrue(torch.isfinite(logits).all())

    def test_official_builder_custom_superset_random_metadata_matches_feature_registry(self):
        selection = _full_custom_selection()
        with _obj_file(OBJ_TETRA) as path:
            sample = build_meshcnn_sample(
                path,
                selection,
                endpoint_order='random',
            )

        self.assertEqual(sample.feature_group, 'custom')
        self.assertEqual(sample.feature_preset, 'custom')
        self.assertEqual(tuple(sample.feature_names), selection.feature_names)
        self.assertEqual(sample.feature_flags, selection.feature_flags.as_dict())
        self.assertEqual(sample.endpoint_order, 'random')
        self.assertEqual(sample.label_source, 'exact_obj')
        self.assertEqual(sample.density_config, selection.density_config)
        self.assertEqual(sample.edge_features.shape[1], len(sample.feature_names))
        for required in ('ao_i', 'ao_j', 'signed_dihedral', 'symmetry_dist', 'density_mean', 'density_diff', 'thickness_sdf'):
            self.assertIn(required, sample.feature_names)

    def test_official_builder_manifest_carries_runtime_slicing_metadata(self):
        selection = _full_custom_selection()
        with _obj_file(OBJ_TETRA) as path:
            sample = build_meshcnn_sample(
                path,
                selection,
                endpoint_order='random',
            )
            manifest = build_dataset_manifest([sample], path.with_suffix('.pt'))

        self.assertEqual(manifest['sample_format'], 'meshcnn_full_v2')
        self.assertEqual(manifest['feature_group'], 'custom')
        self.assertEqual(manifest['feature_preset'], 'custom')
        self.assertEqual(manifest['feature_names'], list(selection.feature_names))
        self.assertEqual(manifest['feature_flags'], selection.feature_flags.as_dict())
        self.assertEqual(manifest['feature_dim'], len(selection.feature_names))
        self.assertEqual(manifest['endpoint_order'], 'random')
        self.assertEqual(manifest['label_source'], 'exact_obj')
        self.assertEqual(manifest['density_config'], selection.density_config)

    def test_slice_meshcnn_dataset_to_control14(self):
        sample = _sample_with_features()
        _, metadata = slice_meshcnn_dataset_features(
            [sample],
            resolve_feature_selection('custom'),
        )

        self.assertEqual(sample.edge_features.shape[1], 14)
        self.assertEqual(tuple(sample.feature_names), PAPER14_FEATURE_NAMES)
        self.assertEqual(metadata['feature_names'], list(PAPER14_FEATURE_NAMES))
        self.assertEqual(metadata['feature_dim'], 14)

    def test_slice_meshcnn_dataset_to_all_optional_features_updates_metadata(self):
        selection = _full_custom_selection()
        sample = _sample_with_features()
        _, metadata = slice_meshcnn_dataset_features([sample], selection)

        self.assertEqual(tuple(sample.feature_names), selection.feature_names)
        self.assertEqual(sample.feature_flags, selection.feature_flags.as_dict())
        self.assertEqual(sample.density_config, selection.density_config)
        self.assertEqual(metadata['feature_names'], list(selection.feature_names))
        self.assertEqual(metadata['feature_flags'], selection.feature_flags.as_dict())
        self.assertEqual(metadata['feature_dim'], len(selection.feature_names))

    def test_slice_meshcnn_dataset_preserves_selected_column_values_in_order(self):
        source = _sample_with_features()
        original = source.edge_features.clone()
        selection = resolve_feature_selection(
            'custom',
            enable_ao=True,
            enable_density=True,
            enable_thickness_sdf=True,
        )
        available = list(source.feature_names)
        expected_indices = torch.tensor([available.index(name) for name in selection.feature_names])

        slice_meshcnn_dataset_features([source], selection)

        self.assertTrue(torch.equal(source.edge_features, original.index_select(1, expected_indices)))

    def test_slice_meshcnn_dataset_missing_feature_raises(self):
        source = _sample_with_features(PAPER14_FEATURE_NAMES)
        selection = resolve_feature_selection('custom', enable_thickness_sdf=True)

        with self.assertRaisesRegex(ValueError, 'thickness_sdf'):
            slice_meshcnn_dataset_features([source], selection)

    def test_slice_meshcnn_dataset_inconsistent_feature_names_raise(self):
        first = _sample_with_features()
        second = _sample_with_features()
        second.feature_names = list(reversed(second.feature_names))

        with self.assertRaisesRegex(ValueError, 'feature_names differ'):
            slice_meshcnn_dataset_features([first, second], resolve_feature_selection('custom'))

    def test_slice_meshcnn_dataset_feature_dim_mismatch_raises(self):
        source = _sample_with_features()
        source.edge_features = source.edge_features[:, :-1]

        with self.assertRaisesRegex(ValueError, 'edge_features dim'):
            slice_meshcnn_dataset_features([source], resolve_feature_selection('custom'))

    def test_runtime_slicing_accepts_official_builder_metadata(self):
        source_selection = _full_custom_selection()
        target_selection = resolve_feature_selection(
            'custom',
            enable_ao=True,
            enable_density=True,
            enable_thickness_sdf=True,
        )
        with _obj_file(OBJ_TETRA) as path:
            sample = build_meshcnn_sample(
                path,
                source_selection,
                endpoint_order='random',
            )
            manifest = build_dataset_manifest([sample], path.with_suffix('.pt'))

        _, metadata = slice_meshcnn_dataset_features([sample], target_selection, manifest)

        self.assertEqual(sample.edge_features.shape[1], len(target_selection.feature_names))
        self.assertEqual(sample.feature_names, list(target_selection.feature_names))
        self.assertEqual(metadata['feature_names'], list(target_selection.feature_names))
        self.assertEqual(metadata['feature_flags'], target_selection.feature_flags.as_dict())
        self.assertEqual(metadata['endpoint_order'], 'random')
        self.assertEqual(metadata['label_source'], 'exact_obj')
        self.assertEqual(metadata['sample_format'], 'meshcnn_full_v2')


if __name__ == '__main__':
    unittest.main()
