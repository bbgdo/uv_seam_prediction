import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from models.utils.seam_topology import apply_topology_pipeline, build_seam_graph_view
from preprocessing.obj_parser import parse_obj
from preprocessing.topology import WeldConfig, build_topology


ROOT = Path(__file__).resolve().parents[1]


def _load_real_mesh_view(name: str):
    mesh = parse_obj(ROOT / '3d-objs_test' / name)
    topology = build_topology(mesh, WeldConfig.exact())
    unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64)
    return build_seam_graph_view(topology, unique_edges), topology


def _edge_index(view, edge: tuple[int, int]) -> int:
    key = tuple(sorted(edge))
    for index, candidate in enumerate(view.unique_edges):
        if (int(candidate[0]), int(candidate[1])) == key:
            return int(index)
    raise AssertionError(f'edge {key} not found')


def _run_truth_pipeline(
    mesh_name: str,
    true_edges: list[tuple[int, int]],
    false_edges: list[tuple[int, int]],
):
    view, topology = _load_real_mesh_view(mesh_name)
    probabilities = np.zeros(view.edge_count, dtype=np.float64)
    anchors: set[int] = set()
    for edge in true_edges:
        probabilities[_edge_index(view, edge)] = 0.95
        anchors.update(int(vertex) for vertex in edge)
    for edge in false_edges:
        probabilities[_edge_index(view, edge)] = 0.0

    result = apply_topology_pipeline(
        view,
        probabilities,
        tau_low=0.5,
        tau_high=0.7,
        d_max=3,
        r_bridge=6,
        l_min=4,
        anchor_boundary=False,
        extra_anchor_vertices=frozenset(anchors),
        topology=topology,
    )
    return view, result


class RealMeshBridgingTests(unittest.TestCase):
    def test_test_triang_truth_edges(self):
        must_bridge = [
            (2558, 2557),
            (1007, 1006),
            (1006, 1988),
            (1988, 2518),
            (5149, 5103),
            (5103, 3005),
            (92, 123),
            (123, 122),
            (4790, 4974),
            (4974, 4975),
            (4975, 5579),
        ]
        must_keep = [
            (2415, 1665),
            (1665, 1666),
            (1666, 1862),
            (4705, 4521),
            (4521, 4522),
            (4522, 5228),
            (5228, 5229),
            (2217, 2222),
            (2222, 2221),
        ]
        must_prune = [
            (4973, 4976),
            (5105, 4541),
            (4541, 4539),
        ]
        view, _ = _load_real_mesh_view('test_triang.obj')
        self.assertEqual(nx.shortest_path_length(view.vertex_graph, 2530, 2544), 13)

        view, result = _run_truth_pipeline('test_triang.obj', must_bridge + must_keep, must_prune)

        for edge in must_bridge + must_keep:
            self.assertTrue(result.final_edge_mask[_edge_index(view, edge)], edge)
        for edge in must_prune:
            self.assertFalse(result.final_edge_mask[_edge_index(view, edge)], edge)
        self.assertGreater(result.bridging_result.steiner_edges_added_total, 0)

    def test_man_test_002_truth_edges(self):
        must_bridge = [
            (2558, 2557),
            (2045, 2541),
            (2541, 4884),
            (5149, 3003),
            (3003, 3005),
            (2994, 2993),
            (2993, 2964),
            (2328, 125),
            (170, 2019),
            (2019, 2018),
            (2018, 2004),
            (5464, 5562),
            (5455, 5477),
            (5477, 5520),
            (5520, 5483),
            (4787, 4169),
            (4169, 4170),
            (4170, 4790),
            (4790, 4974),
        ]
        must_keep = [
            (4705, 4521),
            (4521, 4522),
            (4522, 5228),
            (5228, 5229),
            (2415, 1665),
            (1665, 1666),
            (1666, 1862),
            (2544, 2541),
            (2541, 2540),
            (5553, 5464),
        ]
        must_prune = [
            (4971, 4972),
            (4972, 5580),
            (3218, 3215),
        ]

        view, result = _run_truth_pipeline('man_test_002.obj', must_bridge + must_keep, must_prune)

        for edge in must_bridge + must_keep:
            self.assertTrue(result.final_edge_mask[_edge_index(view, edge)], edge)
        for edge in must_prune:
            self.assertFalse(result.final_edge_mask[_edge_index(view, edge)], edge)
        self.assertGreater(result.bridging_result.steiner_edges_added_total, 0)
