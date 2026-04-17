import unittest
from contextlib import contextmanager
from pathlib import Path
import tempfile

import numpy as np
import torch

from models.meshcnn_full.mesh import MutableMeshTopology, build_mesh_adjacency
from models.meshcnn_full.model import MeshCNNSegmenter
from models.meshcnn_full.pool import MeshPool
from models.meshcnn_full.unpool import MeshUnpool
from preprocessing.build_meshcnn_dataset_v2 import build_meshcnn_sample
from preprocessing.feature_registry import resolve_feature_selection


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


if __name__ == '__main__':
    unittest.main()
