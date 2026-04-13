import unittest

import numpy as np
import torch
import trimesh

from models.dual_graphsage.model import DualGraphSAGE
from preprocessing.compute_features import compute_edge_features


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
    def test_feature_preset_shapes(self):
        mesh = _tiny_mesh()

        paper, edges, _ = compute_edge_features(mesh, feature_preset='paper14', endpoint_order='random')
        extended, extended_edges, _ = compute_edge_features(mesh, feature_preset='extended18')

        self.assertEqual(paper.shape, (len(edges), 14))
        self.assertEqual(extended.shape, (len(extended_edges), 18))

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


if __name__ == '__main__':
    unittest.main()
