import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from models.utils.seam_topology import (
    apply_topology_pipeline,
    build_seam_graph_view,
    compute_topology_preserving_skeleton,
)
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


@dataclass(frozen=True)
class _HonestProbabilityFixture:
    view: Any
    topology: Any
    probabilities: np.ndarray
    true_skeleton_edges: list[tuple[int, int]]
    gap_edges: list[tuple[int, int]]
    false_edges: list[tuple[int, int]]


def _build_honest_fixture(
    mesh_name: str,
    true_skeleton_edges: list[tuple[int, int]],
    gap_edges: list[tuple[int, int]],
    *,
    p_skeleton: float = 0.95,
    p_gap: float = 0.01,
    p_false: float = 0.001,
) -> _HonestProbabilityFixture:
    if p_skeleton < 0.70:
        raise ValueError(f'p_skeleton must be >= 0.70, got {p_skeleton}')
    if p_gap >= 0.30:
        raise ValueError(f'p_gap must be < 0.30, got {p_gap}')
    if p_false >= 0.30:
        raise ValueError(f'p_false must be < 0.30, got {p_false}')

    view, topology = _load_real_mesh_view(mesh_name)

    true_skeleton_keys = {tuple(sorted(edge)) for edge in true_skeleton_edges}
    gap_keys = {tuple(sorted(edge)) for edge in gap_edges}
    all_edge_keys = {
        (int(edge[0]), int(edge[1]))
        for edge in view.unique_edges
    }

    for edge in sorted(true_skeleton_keys | gap_keys):
        if edge not in all_edge_keys:
            raise ValueError(f'edge {edge} not found in mesh {mesh_name}')

    probabilities = np.full(view.edge_count, p_false, dtype=np.float64)
    for edge in true_skeleton_edges:
        probabilities[_edge_index(view, edge)] = p_skeleton
    for edge in gap_edges:
        probabilities[_edge_index(view, edge)] = p_gap

    false_edges = [
        edge
        for edge in sorted(all_edge_keys)
        if edge not in true_skeleton_keys and edge not in gap_keys
    ]
    return _HonestProbabilityFixture(
        view=view,
        topology=topology,
        probabilities=probabilities,
        true_skeleton_edges=list(true_skeleton_edges),
        gap_edges=list(gap_edges),
        false_edges=false_edges,
    )


class RealMeshBridgingTests(unittest.TestCase):
    @staticmethod
    def _run_honest_pipeline(fixture: _HonestProbabilityFixture):
        anchors = frozenset(
            int(vertex)
            for edge in fixture.true_skeleton_edges
            for vertex in edge
        )
        return apply_topology_pipeline(
            fixture.view,
            fixture.probabilities,
            tau_low=0.30,
            d_max=3,
            r_bridge=6,
            l_min=4,
            anchor_boundary=False,
            extra_anchor_vertices=anchors,
            topology=fixture.topology,
        )

    def assert_gap_edges_are_honest_and_bridged(
        self,
        fixture: _HonestProbabilityFixture,
        result,
    ) -> None:
        survived_bridge_edges = 0
        for edge in fixture.gap_edges:
            edge_index = _edge_index(fixture.view, edge)
            self.assertLess(fixture.probabilities[edge_index], 0.30, edge)
            self.assertFalse(result.skeleton_result.skeleton_edge_mask[edge_index], edge)
            self.assertTrue(result.bridging_result.bridged_edge_mask[edge_index], edge)
            self.assertIn(edge_index, result.bridging_result.added_bridge_edges, edge)
            if result.final_edge_mask[edge_index]:
                survived_bridge_edges += 1
        self.assertGreater(survived_bridge_edges, 0)
        self.assertGreaterEqual(
            len(result.bridging_result.accepted_bridge_edge_indices),
            len(fixture.gap_edges),
        )

    def test_fixture_actually_creates_a_gap(self):
        fixture = _build_honest_fixture(
            mesh_name='test_triang.obj',
            true_skeleton_edges=[(1007, 1006), (1988, 2518)],
            gap_edges=[(1006, 1988)],
        )
        skel = compute_topology_preserving_skeleton(
            view=fixture.view,
            probabilities=fixture.probabilities,
            tau_low=0.30,
            d_max=3,
            anchor_boundary=False,
            topology=fixture.topology,
        )
        gap_idx = _edge_index(fixture.view, (1006, 1988))
        self.assertFalse(
            skel.skeleton_edge_mask[gap_idx],
            "Honest fixture invariant violated: gap edge ended up in "
            "the skeleton mask. Either p_gap >= tau_low (test cheating) "
            "or the mesh edge is not actually the gap edge intended."
        )

    def test_honest_bridges_test_triang_edges(self):
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
        path_vertices = nx.shortest_path(view.vertex_graph, 2530, 2544)
        path_edges = [
            (int(path_vertices[index]), int(path_vertices[index + 1]))
            for index in range(len(path_vertices) - 1)
        ]

        gap_edges = [
            (1006, 1988),
            (1369, 2234),
            (4974, 4975),
        ]
        true_skeleton_edges = [
            edge
            for edge in must_bridge + path_edges
            if edge not in gap_edges
        ]
        fixture = _build_honest_fixture(
            'test_triang.obj',
            true_skeleton_edges=true_skeleton_edges + must_keep,
            gap_edges=gap_edges,
        )
        result = self._run_honest_pipeline(fixture)

        self.assert_gap_edges_are_honest_and_bridged(fixture, result)
        for edge in must_bridge:
            self.assertTrue(result.bridging_result.bridged_edge_mask[_edge_index(fixture.view, edge)], edge)
        bridged_graph = nx.Graph()
        for edge_index in np.flatnonzero(result.bridging_result.bridged_edge_mask):
            u, v = fixture.view.unique_edges[int(edge_index)]
            bridged_graph.add_edge(int(u), int(v))
        self.assertTrue(nx.has_path(bridged_graph, 2530, 2544))
        for edge in must_keep:
            self.assertTrue(result.final_edge_mask[_edge_index(fixture.view, edge)], edge)
        for edge in must_prune:
            self.assertFalse(result.final_edge_mask[_edge_index(fixture.view, edge)], edge)
        self.assertGreater(result.bridging_result.bridges_accepted, 0)
        self.assertGreater(result.bridging_result.added_bridge_edges_count, 0)
        self.assertGreater(len(result.bridging_result.accepted_bridge_reports), 0)

    def test_honest_bridges_man_edges(self):
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

        gap_edges = [
            (2019, 2018),
            (5477, 5520),
            (4169, 4170),
        ]
        true_skeleton_edges = [edge for edge in must_bridge if edge not in gap_edges]
        fixture = _build_honest_fixture(
            'man_test_002.obj',
            true_skeleton_edges=true_skeleton_edges + must_keep,
            gap_edges=gap_edges,
        )
        result = self._run_honest_pipeline(fixture)

        self.assert_gap_edges_are_honest_and_bridged(fixture, result)
        for edge in must_bridge:
            self.assertTrue(result.bridging_result.bridged_edge_mask[_edge_index(fixture.view, edge)], edge)
        for edge in must_keep:
            self.assertTrue(result.final_edge_mask[_edge_index(fixture.view, edge)], edge)
        for edge in must_prune:
            self.assertFalse(result.final_edge_mask[_edge_index(fixture.view, edge)], edge)
        self.assertGreater(result.bridging_result.bridges_accepted, 0)
        self.assertGreater(result.bridging_result.added_bridge_edges_count, 0)
        self.assertGreater(len(result.bridging_result.accepted_bridge_reports), 0)
