import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from models.meshcnn_full.mesh import MeshCNNSample
from models.meshcnn_full.sparse_layers import SparseMeshConv, SparseMeshPool, SparseMeshUnpool
from models.meshcnn_full.sparse_model import SparseMeshUNetSegmenter
from models.meshcnn_full.sparse_precompute import (
    assert_sparse_cache_cpu_only,
    build_slot_matrices,
    build_sparse_cache,
    get_or_build_sparse_cache,
    materialize_sparse_cache_for_step,
)


def _unique_edges_from_faces(faces: torch.Tensor) -> torch.Tensor:
    raw = torch.stack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        dim=1,
    ).reshape(-1, 2)
    return torch.unique(torch.sort(raw, dim=1).values, dim=0)


def _toy_sample(fin: int = 15) -> MeshCNNSample:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
    unique_edges = _unique_edges_from_faces(faces)
    edge_count = unique_edges.shape[0]
    return MeshCNNSample(
        vertices=vertices,
        faces=faces,
        unique_edges=unique_edges,
        edge_features=torch.randn(edge_count, fin),
        edge_labels=torch.randint(0, 2, (edge_count,), dtype=torch.float32),
        edge_neighbors=torch.full((edge_count, 4), -1, dtype=torch.long),
        edge_to_faces=torch.full((edge_count, 2), -1, dtype=torch.long),
        face_to_edges=torch.full((faces.shape[0], 3), -1, dtype=torch.long),
        boundary_mask=torch.zeros(edge_count, dtype=torch.bool),
        file_path='toy.obj',
        feature_group='test',
        feature_preset='test',
        feature_names=[f'f{i}' for i in range(fin)],
        feature_flags={},
        endpoint_order='fixed',
    )


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _assert_persistent_cache_cpu_only(testcase: unittest.TestCase, sample: MeshCNNSample) -> None:
    testcase.assertIsInstance(sample.sparse_cache, dict)
    testcase.assertNotIn('_device_caches', sample.sparse_cache)
    testcase.assertNotIn('device_caches', sample.sparse_cache)
    assert_sparse_cache_cpu_only(sample.sparse_cache)
    for tensor in _iter_tensors(sample.sparse_cache):
        testcase.assertEqual(tensor.device.type, 'cpu')


class SparsePrecomputeTests(unittest.TestCase):
    def test_build_slot_matrices_interior_edge(self):
        faces = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
        unique_edges = _unique_edges_from_faces(faces)
        slots = build_slot_matrices(unique_edges, faces)

        edge_to_id = {tuple(edge.tolist()): idx for idx, edge in enumerate(unique_edges)}
        shared = edge_to_id[(0, 1)]
        expected = [
            edge_to_id[(1, 2)],
            edge_to_id[(0, 2)],
            edge_to_id[(0, 3)],
            edge_to_id[(1, 3)],
        ]

        for slot, neighbor in zip(slots, expected):
            dense = slot.to_dense()
            self.assertEqual(float(dense[shared, neighbor]), 1.0)
            self.assertEqual(float(dense[shared].sum()), 1.0)

    def test_build_slot_matrices_boundary_edge(self):
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        unique_edges = _unique_edges_from_faces(faces)
        slots = build_slot_matrices(unique_edges, faces)
        edge_to_id = {tuple(edge.tolist()): idx for idx, edge in enumerate(unique_edges)}
        boundary = edge_to_id[(0, 1)]

        self.assertEqual(float(slots[0].to_dense()[boundary].sum()), 1.0)
        self.assertEqual(float(slots[1].to_dense()[boundary].sum()), 1.0)
        self.assertEqual(float(slots[2].to_dense()[boundary].sum()), 0.0)
        self.assertEqual(float(slots[3].to_dense()[boundary].sum()), 0.0)

    def test_pool_map_properties(self):
        sample = _toy_sample()
        cache = build_sparse_cache(sample, pool_ratios=(0.6, 0.4), min_edges_per_level=1)

        for pool_map, unpool_map in zip(cache['pool_maps'], cache['unpool_maps']):
            dense = pool_map.to_dense()
            self.assertEqual(pool_map.shape[1], unpool_map.shape[0])
            self.assertEqual(pool_map.shape[0], unpool_map.shape[1])
            self.assertTrue(torch.allclose(dense.sum(dim=0), torch.ones(dense.shape[1])))
            self.assertTrue(torch.all(dense.sum(dim=1) > 0))

    def test_coarse_slot_shapes(self):
        sample = _toy_sample()
        cache = build_sparse_cache(sample, pool_ratios=(0.6,), min_edges_per_level=1)

        edge_counts = cache['edge_counts']
        for level, slots in enumerate(cache['slot_adj_levels']):
            for slot in slots:
                self.assertEqual(slot.shape, (edge_counts[level], edge_counts[level]))
                row_sums = torch.sparse.sum(slot, dim=1).to_dense()
                non_empty = row_sums > 0
                self.assertTrue(torch.all(row_sums <= 1.0 + 1e-6))
                self.assertTrue(torch.allclose(row_sums[non_empty], torch.ones_like(row_sums[non_empty])))

    def test_persistent_sparse_cache_is_cpu_only(self):
        sample = _toy_sample()
        cache = get_or_build_sparse_cache(sample, pool_ratios=(0.6, 0.4), min_edges_per_level=1)

        self.assertIs(cache, sample.sparse_cache)
        _assert_persistent_cache_cpu_only(self, sample)

    def test_materialized_step_cache_is_not_stored_back_into_sample(self):
        sample = _toy_sample()
        cpu_cache = get_or_build_sparse_cache(sample, pool_ratios=(0.6, 0.4), min_edges_per_level=1)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        step_cache = materialize_sparse_cache_for_step(cpu_cache, device)

        _assert_persistent_cache_cpu_only(self, sample)
        for cpu_tensor, step_tensor in zip(_iter_tensors(cpu_cache), _iter_tensors(step_cache)):
            self.assertIsNot(cpu_tensor, step_tensor)
            self.assertEqual(step_tensor.device, device)


class SparseLayerTests(unittest.TestCase):
    def test_sparse_meshconv_output_shape(self):
        sample = _toy_sample(fin=7)
        slots = build_slot_matrices(sample.unique_edges, sample.faces)
        conv = SparseMeshConv(7, 11)
        out = conv(sample.edge_features, slots)
        self.assertEqual(out.shape, (sample.num_edges, 11))

    def test_sparse_meshconv_face_order_invariance(self):
        sample = _toy_sample(fin=5)
        slots = build_slot_matrices(sample.unique_edges, sample.faces)
        swapped_slots = (slots[2], slots[3], slots[0], slots[1])
        conv = SparseMeshConv(5, 9)

        out = conv(sample.edge_features, slots)
        swapped = conv(sample.edge_features, swapped_slots)
        self.assertTrue(torch.allclose(out, swapped, atol=1e-6))

    def test_sparse_pool_matches_manual_weighted_average(self):
        indices = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long)
        values = torch.ones(3)
        pool_map = torch.sparse_coo_tensor(indices, values, (2, 3)).coalesce()
        x = torch.tensor([[1.0, 2.0], [3.0, 6.0], [10.0, 20.0]])
        pool = SparseMeshPool(2)
        with torch.no_grad():
            for param in pool.gate.parameters():
                param.zero_()

        out = pool(x, pool_map)
        expected = torch.tensor([[2.0, 4.0], [10.0, 20.0]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_sparse_unpool_matches_transpose_broadcast(self):
        indices = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
        values = torch.ones(3)
        unpool_map = torch.sparse_coo_tensor(indices, values, (3, 2)).coalesce()
        coarse = torch.tensor([[2.0, 4.0], [10.0, 20.0]])

        out = SparseMeshUnpool()(coarse, unpool_map)
        expected = torch.tensor([[2.0, 4.0], [2.0, 4.0], [10.0, 20.0]])
        self.assertTrue(torch.allclose(out, expected))


class SparseModelTests(unittest.TestCase):
    def test_model_accepts_arbitrary_in_channels(self):
        for fin in (5, 15, 23):
            sample = _toy_sample(fin=fin)
            model = SparseMeshUNetSegmenter(
                in_channels=fin,
                hidden_channels=8,
                pool_ratios=(0.7, 0.5),
                min_edges=1,
            )
            logits = model(sample)
            self.assertEqual(logits.shape, sample.edge_labels.shape)

    def test_model_returns_logits_for_original_edges(self):
        sample = _toy_sample(fin=15)
        model = SparseMeshUNetSegmenter(
            in_channels=15,
            hidden_channels=8,
            pool_ratios=(0.7, 0.5),
            min_edges=1,
        )
        logits = model(sample)
        self.assertEqual(logits.shape[0], sample.unique_edges.shape[0])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA large-mesh smoke requires a GPU')
    def test_backward_large_mesh_smoke(self):
        grid = 82
        vertices = []
        faces = []
        for y in range(grid + 1):
            for x in range(grid + 1):
                vertices.append([float(x), float(y), 0.0])
        for y in range(grid):
            for x in range(grid):
                a = y * (grid + 1) + x
                b = a + 1
                c = a + grid + 1
                d = c + 1
                faces.append([a, b, d])
                faces.append([a, d, c])
        faces = torch.tensor(faces, dtype=torch.long)
        unique_edges = _unique_edges_from_faces(faces)
        edge_count = unique_edges.shape[0]
        sample = MeshCNNSample(
            vertices=torch.tensor(vertices, dtype=torch.float32),
            faces=faces,
            unique_edges=unique_edges,
            edge_features=torch.randn(edge_count, 15),
            edge_labels=torch.randint(0, 2, (edge_count,), dtype=torch.float32),
            edge_neighbors=torch.full((edge_count, 4), -1, dtype=torch.long),
            edge_to_faces=torch.full((edge_count, 2), -1, dtype=torch.long),
            face_to_edges=torch.full((faces.shape[0], 3), -1, dtype=torch.long),
            boundary_mask=torch.zeros(edge_count, dtype=torch.bool),
            file_path='grid.obj',
            feature_group='test',
            feature_preset='test',
            feature_names=[f'f{i}' for i in range(15)],
            feature_flags={},
            endpoint_order='fixed',
        )
        self.assertGreaterEqual(edge_count, 20000)
        model = SparseMeshUNetSegmenter(
            in_channels=15,
            hidden_channels=16,
            pool_ratios=(0.8, 0.6),
            min_edges=32,
        ).cuda()
        logits = model(sample)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            sample.edge_labels.cuda(),
        )
        loss.backward()
        self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_no_dense_fallback_in_hot_path(self):
        sample = _toy_sample(fin=15)
        model = SparseMeshUNetSegmenter(
            in_channels=15,
            hidden_channels=8,
            pool_ratios=(0.7, 0.5),
            min_edges=1,
        )
        build_sparse_cache(sample, pool_ratios=(0.7, 0.5), min_edges_per_level=1)

        with mock.patch.object(torch.Tensor, 'to_dense', side_effect=AssertionError('to_dense called')):
            logits = model(sample)

        self.assertEqual(logits.shape, sample.edge_labels.shape)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA sparse-cache lifecycle test requires a GPU')
    def test_forward_does_not_persist_cuda_cache(self):
        sample = _toy_sample(fin=15)
        model = SparseMeshUNetSegmenter(
            in_channels=15,
            hidden_channels=8,
            pool_ratios=(0.7, 0.5),
            min_edges=1,
        ).cuda()

        logits = model(sample)

        self.assertEqual(logits.device.type, 'cuda')
        _assert_persistent_cache_cpu_only(self, sample)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA sparse-cache lifecycle test requires a GPU')
    def test_repeated_two_sample_forward_does_not_accumulate_persistent_cuda_refs(self):
        sample_a = _toy_sample(fin=15)
        sample_b = _toy_sample(fin=15)
        model = SparseMeshUNetSegmenter(
            in_channels=15,
            hidden_channels=8,
            pool_ratios=(0.7, 0.5),
            min_edges=1,
        ).cuda()

        _ = model(sample_a)
        _assert_persistent_cache_cpu_only(self, sample_a)
        self.assertIsNone(sample_b.sparse_cache)

        _ = model(sample_b)
        _assert_persistent_cache_cpu_only(self, sample_a)
        _assert_persistent_cache_cpu_only(self, sample_b)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA two-step training smoke requires a GPU')
    def test_training_two_steps_smoke_on_gpu(self):
        samples = [_toy_sample(fin=15), _toy_sample(fin=15)]
        model = SparseMeshUNetSegmenter(
            in_channels=15,
            hidden_channels=8,
            pool_ratios=(0.7, 0.5),
            min_edges=1,
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for sample in samples:
            optimizer.zero_grad(set_to_none=True)
            logits = model(sample)
            loss = F.binary_cross_entropy_with_logits(logits, sample.edge_labels.cuda())
            loss.backward()
            optimizer.step()
            for seen in samples:
                if seen.sparse_cache is not None:
                    _assert_persistent_cache_cpu_only(self, seen)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA memory smoke requires a GPU')
    def test_max_memory_stabilizes_across_steps(self):
        device = torch.device('cuda')
        samples = [_toy_sample(fin=15), _toy_sample(fin=15), _toy_sample(fin=15)]
        model = SparseMeshUNetSegmenter(
            in_channels=15,
            hidden_channels=8,
            pool_ratios=(0.7, 0.5),
            min_edges=1,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        after_step_allocated: list[int] = []

        for sample in samples:
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(sample)
            labels = sample.edge_labels.to(device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            self.assertGreater(torch.cuda.max_memory_allocated(device), torch.cuda.memory_allocated(device) - 1)
            loss.backward()
            optimizer.step()
            del logits, labels, loss
            torch.cuda.synchronize(device)
            after_step_allocated.append(torch.cuda.memory_allocated(device))
            _assert_persistent_cache_cpu_only(self, sample)

        persistent_growth = max(after_step_allocated[1:]) - after_step_allocated[1]
        self.assertLess(persistent_growth, 16 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main()
