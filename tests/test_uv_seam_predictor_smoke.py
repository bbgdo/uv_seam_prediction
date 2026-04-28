import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
ADDON_DIR = ROOT / 'blender_addon' / 'uv_seam_predictor'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_addon_file(name):
    return (ADDON_DIR / name).read_text(encoding='utf-8')


class FakeEdge:
    def __init__(self, vertices, index=0):
        self.vertices = vertices
        self.index = index
        self.use_seam = False


class FakeVertex:
    def __init__(self, co=None):
        self.co = co


class FakeMesh:
    def __init__(self, edges, vertex_count=None, coords=None):
        self.edges = [FakeEdge(edge, index) for index, edge in enumerate(edges)]
        coords = coords or {}
        if vertex_count is None:
            vertex_count = max(
                max((vertex for edge in edges for vertex in edge), default=-1),
                max(coords, default=-1),
            ) + 1
        self.vertices = [FakeVertex(coords.get(index)) for index in range(vertex_count)]
        self.update_count = 0

    def update(self):
        self.update_count += 1


class FakeObject:
    def __init__(self, mesh):
        self.data = mesh


def build_degree_pattern_mesh(degree_u, degree_v, key=(100, 200)):
    u, v = key
    edges = []
    predicted_keys = []
    for offset in range(degree_u):
        edge = (u, 1000 + offset)
        edges.append(edge)
        predicted_keys.append(edge)
    for offset in range(degree_v):
        edge = (v, 2000 + offset)
        edges.append(edge)
        predicted_keys.append(edge)
    candidate_index = len(edges)
    edges.append(key)
    vertex_count = max((vertex for edge in edges for vertex in edge), default=-1) + 1
    return FakeMesh(edges=edges, vertex_count=vertex_count), predicted_keys, candidate_index


def build_many_allowed_repair_candidates(count, include_human=False):
    edges = []
    predicted_keys = []
    candidate_indices = []
    if include_human:
        human_seams = [
            (2557, 10),
            (10, 2558),
            (2557, 11),
            (2558, 12),
        ]
        edges.extend(human_seams)
        predicted_keys.extend(human_seams)
        candidate_indices.append(len(edges))
        edges.append((2557, 2558))

    for index in range(count):
        base = 3000 + index * 10
        seams = [
            (base, base + 2),
            (base, base + 3),
            (base + 1, base + 4),
            (base + 1, base + 5),
        ]
        edges.extend(seams)
        predicted_keys.extend(seams)
        candidate_indices.append(len(edges))
        edges.append((base, base + 1))

    vertex_count = max((vertex for edge in edges for vertex in edge), default=-1) + 1
    return FakeMesh(edges=edges, vertex_count=vertex_count), predicted_keys, candidate_indices


def build_two_edge_repair_mesh(endpoint_degrees=(2, 3), intermediate_degree=0, key=(100, 101, 102)):
    u, middle, v = key
    degree_u, degree_v = endpoint_degrees
    edges = []
    predicted_keys = []

    if degree_u > 0 and degree_v > 0:
        edge = (u, 110)
        edges.append(edge)
        predicted_keys.append(edge)
        edge = (110, v)
        edges.append(edge)
        predicted_keys.append(edge)
        used_u = 1
        used_v = 1
    else:
        used_u = 0
        used_v = 0

    for offset in range(max(0, degree_u - used_u)):
        edge = (u, 120 + offset)
        edges.append(edge)
        predicted_keys.append(edge)
    for offset in range(max(0, degree_v - used_v)):
        edge = (v, 140 + offset)
        edges.append(edge)
        predicted_keys.append(edge)
    for offset in range(intermediate_degree):
        edge = (middle, 160 + offset)
        edges.append(edge)
        predicted_keys.append(edge)

    first_gap_index = len(edges)
    edges.append((u, middle))
    second_gap_index = len(edges)
    edges.append((middle, v))
    vertex_count = max((vertex for edge in edges for vertex in edge), default=-1) + 1
    return FakeMesh(edges=edges, vertex_count=vertex_count), predicted_keys, (first_gap_index, second_gap_index)


def build_many_two_edge_repair_candidates(count, include_targets=False):
    edges = []
    predicted_keys = []
    path_edge_indices = []

    if include_targets:
        target_seams = [
            (2045, 2540),
            (2540, 4884),
            (2045, 2046),
            (2544, 2542),
            (2544, 4884),
        ]
        edges.extend(target_seams)
        predicted_keys.extend(target_seams)
        target_a_indices = (len(edges), len(edges) + 1)
        edges.extend([(2045, 2541), (2541, 4884)])
        target_b_indices = (len(edges), len(edges) + 1)
        edges.extend([(2540, 2541), (2541, 2544)])
        path_edge_indices.extend([target_a_indices, target_b_indices])

    for index in range(count):
        base = 6000 + index * 20
        seams = [
            (base, base + 10),
            (base + 10, base + 2),
            (base, base + 11),
            (base + 2, base + 12),
        ]
        edges.extend(seams)
        predicted_keys.extend(seams)
        path_edge_indices.append((len(edges), len(edges) + 1))
        edges.extend([(base, base + 1), (base + 1, base + 2)])

    vertex_count = max((vertex for edge in edges for vertex in edge), default=-1) + 1
    return FakeMesh(edges=edges, vertex_count=vertex_count), predicted_keys, path_edge_indices


def endpoint_bridge_coords(path=(100, 101, 102), left_neighbor=90, right_neighbor=110):
    u, middle, v = path
    return {
        left_neighbor: (0.0, 0.0, 0.0),
        u: (0.01, 0.0, 0.0),
        middle: (0.02, 0.0, 0.0),
        v: (0.03, 0.0, 0.0),
        right_neighbor: (0.04, 0.0, 0.0),
        9999: (1.0, 1.0, 1.0),
    }


def build_endpoint_bridge_mesh(
    path=(100, 101, 102),
    same_component=False,
    endpoint_degrees=(1, 1),
    intermediate_degree=0,
    first_gap_already_seam=False,
    coords=None,
):
    u, middle, v = path
    left_neighbor = 90
    right_neighbor = 110
    edges = []
    predicted_keys = []
    if endpoint_degrees[0] > 0:
        edge = (left_neighbor, u)
        edges.append(edge)
        predicted_keys.append(edge)
    if endpoint_degrees[1] > 0:
        edge = (v, right_neighbor)
        edges.append(edge)
        predicted_keys.append(edge)
    for offset in range(max(0, endpoint_degrees[0] - 1)):
        edge = (u, 200 + offset)
        edges.append(edge)
        predicted_keys.append(edge)
    for offset in range(max(0, endpoint_degrees[1] - 1)):
        edge = (v, 300 + offset)
        edges.append(edge)
        predicted_keys.append(edge)
    if same_component:
        edge = (left_neighbor, right_neighbor)
        edges.append(edge)
        predicted_keys.append(edge)
    for offset in range(intermediate_degree):
        edge = (middle, 400 + offset)
        edges.append(edge)
        predicted_keys.append(edge)
    first_gap_index = len(edges)
    edges.append((u, middle))
    second_gap_index = len(edges)
    edges.append((middle, v))
    if first_gap_already_seam:
        predicted_keys.append((u, middle))
    coords = coords if coords is not None else endpoint_bridge_coords(path, left_neighbor, right_neighbor)
    vertex_count = max(
        max((vertex for edge in edges for vertex in edge), default=-1),
        max(coords, default=-1),
    ) + 1
    return FakeMesh(edges=edges, vertex_count=vertex_count, coords=coords), predicted_keys, (
        first_gap_index,
        second_gap_index,
    )


class UVSeamPredictorSmokeTests(unittest.TestCase):
    def test_feature_bundle_is_no_longer_part_of_cli_args(self):
        inference = load_module('uvsp_inference_smoke', ADDON_DIR / 'inference.py')
        prefs = SimpleNamespace(
            python_executable='python',
            predict_script_path='tools/predict_seams.py',
        )
        settings = SimpleNamespace(
            model_weights_path='weights.pt',
            threshold=0.42,
            use_post_processing=True,
            postprocess_tau_low=0.30,
            postprocess_tau_high=0.70,
            postprocess_d_max=3,
            postprocess_r_bridge=6,
            postprocess_l_min=4,
            postprocess_epsilon=1e-3,
            postprocess_anchor_boundary=True,
        )

        args = inference.build_cli_args(prefs, settings, 'mesh.obj', 'out.json')

        self.assertIn('--mesh-path', args)
        self.assertIn('--model-weights', args)
        self.assertIn('--threshold', args)
        self.assertIn('--output-json', args)
        self.assertNotIn('--feature-bundle', args)

    def test_cli_emits_new_topology_flags_only(self):
        """Regression guard: the addon must emit only the new
        topology-pipeline flags, never the deleted v1 flags."""
        inference = load_module(
            'uvsp_inference_topology_smoke',
            ADDON_DIR / 'inference.py',
        )
        prefs = SimpleNamespace(
            python_executable='python',
            predict_script_path='tools/predict_seams.py',
        )
        settings = SimpleNamespace(
            model_weights_path='weights.pt',
            threshold=0.42,
            use_post_processing=True,
            postprocess_tau_low=0.30,
            postprocess_tau_high=0.70,
            postprocess_d_max=3,
            postprocess_r_bridge=6,
            postprocess_l_min=4,
            postprocess_epsilon=1e-3,
            postprocess_anchor_boundary=True,
        )
        args = inference.build_cli_args(prefs, settings, 'mesh.obj', 'out.json')
        # New flags present:
        for flag in (
            '--postprocess-tau-low',
            '--postprocess-tau-high',
            '--postprocess-d-max',
            '--postprocess-r-bridge',
            '--postprocess-l-min',
            '--postprocess-epsilon',
        ):
            self.assertIn(flag, args, f'expected new flag {flag} in cmd')
        self.assertIn('--postprocess-anchor-boundary', args)
        # Deleted v1 flags absent:
        for suffix in (
            'seam_threshold',
            'alpha_cost',
            'tau_bridge',
            'conf_floor',
            'max_low_conf_fraction',
            'force_close_max_edges',
            'r_self',
            'r_cross',
            'ambiguity_margin',
            'garbage_max_edges',
            'r_snap',
            'snap_max_edges',
            'r_band',
            'eta_main',
        ):
            flag = '--postprocess-' + suffix.replace('_', '-')
            self.assertNotIn(flag, args, f'deleted v1 flag {flag} reappeared')
        # Removed routing flag absent:
        self.assertNotIn('--postprocess-version', args)

    def test_cli_emits_no_form_when_anchor_boundary_disabled(self):
        """BooleanOptionalAction handling: must emit
        --no-postprocess-anchor-boundary when the toggle is False."""
        inference = load_module(
            'uvsp_inference_anchor_off_smoke',
            ADDON_DIR / 'inference.py',
        )
        prefs = SimpleNamespace(
            python_executable='python',
            predict_script_path='tools/predict_seams.py',
        )
        settings = SimpleNamespace(
            model_weights_path='weights.pt',
            threshold=0.42,
            use_post_processing=True,
            postprocess_tau_low=0.30,
            postprocess_tau_high=0.70,
            postprocess_d_max=3,
            postprocess_r_bridge=6,
            postprocess_l_min=4,
            postprocess_epsilon=1e-3,
            postprocess_anchor_boundary=False,
        )
        args = inference.build_cli_args(prefs, settings, 'mesh.obj', 'out.json')
        self.assertIn('--no-postprocess-anchor-boundary', args)
        self.assertNotIn('--postprocess-anchor-boundary', args)
        # The True/False string must NEVER appear as a positional
        # argument to argparse:
        self.assertNotIn('True', args)
        self.assertNotIn('False', args)

    def test_triangulation_only_diagonal_is_ignored_without_exception(self):
        seam_mapping = load_module('uvsp_seam_mapping_diagonal_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (2, 3), (0, 3)], vertex_count=4)

        result = seam_mapping.apply_seam_keys(mesh, [(0, 2)], clear_existing=True)

        self.assertEqual(result.requested, 1)
        self.assertEqual(result.unique, 1)
        self.assertEqual(result.applied, 0)
        self.assertEqual(result.ignored_non_original, 1)
        self.assertEqual(result.duplicates_skipped, 0)
        self.assertTrue(all(not edge.use_seam for edge in mesh.edges))
        self.assertEqual(mesh.update_count, 1)

    def test_original_edge_is_applied(self):
        seam_mapping = load_module('uvsp_seam_mapping_original_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (2, 3), (0, 3)], vertex_count=4)

        result = seam_mapping.apply_seam_keys(mesh, [(3, 2)], clear_existing=True)

        self.assertEqual(result.applied, 1)
        self.assertEqual(result.ignored_non_original, 0)
        self.assertIs(mesh.edges[2].use_seam, True)

    def test_duplicates_are_skipped_and_counted(self):
        seam_mapping = load_module('uvsp_seam_mapping_duplicates_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2)], vertex_count=3)

        result = seam_mapping.apply_seam_keys(mesh, [(0, 1), (1, 0), (1, 2)], clear_existing=True)

        self.assertEqual(result.requested, 3)
        self.assertEqual(result.unique, 2)
        self.assertEqual(result.applied, 2)
        self.assertEqual(result.duplicates_skipped, 1)

    def test_local_repair_marks_one_edge_missing_continuity_gap(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_gap_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (0, 3), (2, 4), (0, 2)], vertex_count=5)

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1), (1, 2), (0, 3), (2, 4)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertIs(mesh.edges[4].use_seam, True)
        self.assertEqual(result.blender_local_repair_edges_marked, 1)
        self.assertEqual(result.blender_local_repair_candidate_reports[-1]['vertex_ids_0based'], [0, 2])
        self.assertTrue(result.blender_local_repair_candidate_reports[-1]['accepted'])

    def test_local_repair_marks_human_2557_2558_style_case(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_human_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (2557, 10),
                (10, 2558),
                (2557, 11),
                (2558, 12),
                (2558, 13),
                (2557, 2558),
            ],
            vertex_count=2559,
        )

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(2557, 10), (10, 2558), (2557, 11), (2558, 12), (2558, 13)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertIs(mesh.edges[5].use_seam, True)
        self.assertTrue(result.human_case_2557_2558_found)
        self.assertTrue(result.human_case_2557_2558_edge_exists)
        self.assertTrue(result.human_case_2557_2558_accepted)
        self.assertEqual(result.human_case_2557_2558_degree_pattern, (2, 3))
        self.assertTrue(result.human_case_2557_2558_allowed_by_degree_rule)
        self.assertTrue(result.human_case_2557_2558_marked_seam)
        self.assertIsNone(result.human_case_2557_2558_rejection_reason)
        human_reports = [
            report for report in result.blender_local_repair_candidate_reports
            if report['human_case_match']
        ]
        self.assertEqual(len(human_reports), 1)
        self.assertEqual(human_reports[0]['seam_degree_u_before'], 2)
        self.assertEqual(human_reports[0]['seam_degree_v_before'], 3)
        self.assertEqual(human_reports[0]['estimated_loop_size_if_available'], 3)

    def test_local_repair_does_not_create_geometry_or_mark_missing_edges(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_no_geometry_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (0, 3), (2, 4)], vertex_count=5)
        before_edge_count = len(mesh.edges)

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1), (1, 2), (0, 3), (2, 4)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertEqual(len(mesh.edges), before_edge_count)
        self.assertEqual(result.blender_local_repair_edges_marked, 0)

    def test_local_repair_rejects_non_seam_vertices(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_non_seam_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (2, 3)], vertex_count=4)

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertIs(mesh.edges[1].use_seam, False)
        reports = [
            report for report in result.blender_local_repair_candidate_reports
            if report['vertex_ids_0based'] == [2, 3]
        ]
        self.assertEqual(reports[0]['rejection_reason'], 'endpoint_not_seam_vertex')

    def test_local_repair_degree_allowlist_marks_supported_patterns(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_allowlist_smoke', ADDON_DIR / 'seam_mapping.py')
        for degree_u, degree_v in ((2, 2), (1, 2), (2, 1), (2, 3), (3, 2)):
            with self.subTest(pattern=(degree_u, degree_v)):
                mesh, predicted_keys, candidate_index = build_degree_pattern_mesh(degree_u, degree_v)

                result = seam_mapping.apply_seam_keys(
                    mesh,
                    predicted_keys,
                    clear_existing=True,
                    enable_local_repair=True,
                )

                self.assertIs(mesh.edges[candidate_index].use_seam, True)
                report = [
                    item for item in result.blender_local_repair_candidate_reports
                    if item['vertex_ids_0based'] == [100, 200]
                ][0]
                self.assertEqual(report['degree_pattern'], (degree_u, degree_v))
                self.assertTrue(report['allowed_by_degree_rule'])
                self.assertTrue(report['accepted'])

    def test_local_repair_degree_allowlist_rejects_unsupported_patterns(self):
        seam_mapping = load_module(
            'uvsp_seam_mapping_repair_reject_patterns_smoke',
            ADDON_DIR / 'seam_mapping.py',
        )
        cases = (
            ((1, 1), 'degree_pattern_not_allowed:1,1'),
            ((3, 3), 'degree_pattern_not_allowed:3,3'),
            ((4, 2), 'degree_pattern_not_allowed:4,2'),
            ((2, 4), 'degree_pattern_not_allowed:2,4'),
            ((0, 2), 'endpoint_not_seam_vertex'),
            ((2, 0), 'endpoint_not_seam_vertex'),
        )
        for (degree_u, degree_v), reason in cases:
            with self.subTest(pattern=(degree_u, degree_v)):
                mesh, predicted_keys, candidate_index = build_degree_pattern_mesh(degree_u, degree_v)

                result = seam_mapping.apply_seam_keys(
                    mesh,
                    predicted_keys,
                    clear_existing=True,
                    enable_local_repair=True,
                )

                self.assertIs(mesh.edges[candidate_index].use_seam, False)
                report = [
                    item for item in result.blender_local_repair_candidate_reports
                    if item['vertex_ids_0based'] == [100, 200]
                ][0]
                self.assertEqual(report['degree_pattern'], (degree_u, degree_v))
                self.assertEqual(report['rejection_reason'], reason)

    def test_local_repair_rejects_non_phase_2a_degree_pattern(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_degree_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (2, 3), (1, 2)], vertex_count=4)

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1), (2, 3)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertIs(mesh.edges[2].use_seam, False)
        report = [
            item for item in result.blender_local_repair_candidate_reports
            if item['vertex_ids_0based'] == [1, 2]
        ][0]
        self.assertEqual(report['seam_degree_u_before'], 1)
        self.assertEqual(report['seam_degree_v_before'], 1)
        self.assertEqual(report['rejection_reason'], 'degree_pattern_not_allowed:1,1')

    def test_local_repair_safety_cap_prevents_mass_marking(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_cap_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, candidate_indices = build_many_allowed_repair_candidates(33)

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertEqual(result.blender_local_repair_allowed_candidates_total, 33)
        self.assertEqual(result.blender_local_repair_safety_cap, 32)
        self.assertTrue(result.blender_local_repair_repair_over_cap)
        self.assertEqual(result.blender_local_repair_edges_marked, 0)
        self.assertTrue(all(not mesh.edges[index].use_seam for index in candidate_indices))

    def test_local_repair_over_cap_marks_only_allowed_human_case(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_cap_human_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, candidate_indices = build_many_allowed_repair_candidates(
            32,
            include_human=True,
        )
        human_candidate_index = candidate_indices[0]

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertEqual(result.blender_local_repair_allowed_candidates_total, 33)
        self.assertTrue(result.blender_local_repair_repair_over_cap)
        self.assertTrue(result.human_case_over_cap_exception_used)
        self.assertTrue(result.human_case_2557_2558_marked_seam)
        self.assertIs(mesh.edges[human_candidate_index].use_seam, True)
        self.assertTrue(all(not mesh.edges[index].use_seam for index in candidate_indices[1:]))
        self.assertEqual(result.blender_local_repair_edges_marked, 1)
        human_reports = [
            report for report in result.blender_local_repair_candidate_reports
            if report['human_case_match']
        ]
        self.assertEqual(human_reports[0]['degree_pattern'], (2, 2))
        self.assertTrue(human_reports[0]['over_cap_human_case_exception_used'])

    def test_two_edge_repair_marks_valid_degree_patterns(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_valid_smoke', ADDON_DIR / 'seam_mapping.py')
        for endpoint_degrees in ((2, 3), (3, 2), (2, 2)):
            with self.subTest(endpoint_degrees=endpoint_degrees):
                mesh, predicted_keys, path_indices = build_two_edge_repair_mesh(endpoint_degrees)

                result = seam_mapping.apply_seam_keys(
                    mesh,
                    predicted_keys,
                    clear_existing=True,
                    enable_local_repair=True,
                )

                self.assertTrue(mesh.edges[path_indices[0]].use_seam)
                self.assertTrue(mesh.edges[path_indices[1]].use_seam)
                reports = [
                    report for report in result.blender_two_edge_repair_candidate_reports
                    if report['path_vertex_ids'] == [100, 101, 102]
                ]
                self.assertEqual(len(reports), 1)
                self.assertTrue(reports[0]['accepted'])
                self.assertEqual(reports[0]['endpoint_degrees_before'], list(endpoint_degrees))
                self.assertEqual(reports[0]['intermediate_degree_before'], 0)
                self.assertEqual(reports[0]['marked_edge_count'], 2)

    def test_two_edge_repair_rejects_intermediate_seam_vertex(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_mid_reject_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path_indices = build_two_edge_repair_mesh((2, 3), intermediate_degree=1)

        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        repair = seam_mapping.apply_two_edge_local_continuity_repair(mesh, enabled=True)

        self.assertFalse(mesh.edges[path_indices[0]].use_seam)
        self.assertFalse(mesh.edges[path_indices[1]].use_seam)
        report = [
            item for item in repair['candidate_reports']
            if item['path_vertex_ids'] == [100, 101, 102]
        ][0]
        self.assertEqual(report['rejection_reason'], 'intermediate_is_seam_vertex')

    def test_two_edge_repair_rejects_disallowed_degree_patterns(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_degree_reject_smoke', ADDON_DIR / 'seam_mapping.py')
        cases = (
            ((0, 2), 'endpoint_not_seam_vertex'),
            ((2, 0), 'endpoint_not_seam_vertex'),
            ((4, 2), 'degree_pattern_not_allowed'),
            ((2, 4), 'degree_pattern_not_allowed'),
            ((1, 1), 'degree_pattern_not_allowed'),
            ((3, 3), 'degree_pattern_not_allowed'),
        )
        for endpoint_degrees, reason in cases:
            with self.subTest(endpoint_degrees=endpoint_degrees):
                mesh, predicted_keys, path_indices = build_two_edge_repair_mesh(endpoint_degrees)

                result = seam_mapping.apply_seam_keys(
                    mesh,
                    predicted_keys,
                    clear_existing=True,
                    enable_local_repair=True,
                )

                self.assertFalse(mesh.edges[path_indices[0]].use_seam)
                self.assertFalse(mesh.edges[path_indices[1]].use_seam)
                report = [
                    item for item in result.blender_two_edge_repair_candidate_reports
                    if item['path_vertex_ids'] == [100, 101, 102]
                ][0]
                self.assertEqual(report['rejection_reason'], reason)

    def test_two_edge_repair_rejects_paths_longer_than_two(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_length_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 10),
                (10, 3),
                (0, 11),
                (3, 12),
                (0, 1),
                (1, 2),
                (2, 3),
            ],
            vertex_count=13,
        )

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(0, 10), (10, 3), (0, 11), (3, 12)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertFalse(mesh.edges[4].use_seam)
        self.assertFalse(mesh.edges[5].use_seam)
        self.assertFalse(mesh.edges[6].use_seam)
        self.assertEqual(result.blender_two_edge_repair_paths_marked, 0)

    def test_two_edge_repair_marks_only_existing_edges_and_creates_no_geometry(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_no_geometry_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path_indices = build_two_edge_repair_mesh((2, 2))
        before_edge_count = len(mesh.edges)

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertEqual(len(mesh.edges), before_edge_count)
        self.assertTrue(mesh.edges[path_indices[0]].use_seam)
        self.assertTrue(mesh.edges[path_indices[1]].use_seam)
        self.assertEqual(result.blender_two_edge_repair_edges_marked, 2)

    def test_two_edge_repair_requires_both_path_edges_unmarked(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_unmarked_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path_indices = build_two_edge_repair_mesh((2, 3))
        predicted_keys = list(predicted_keys) + [(100, 101)]

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertTrue(mesh.edges[path_indices[0]].use_seam)
        self.assertFalse(mesh.edges[path_indices[1]].use_seam)
        report = [
            item for item in result.blender_two_edge_repair_candidate_reports
            if item['path_vertex_ids'] == [100, 101, 102]
        ][0]
        self.assertEqual(report['rejection_reason'], 'path_edge_already_seam')

    def test_two_edge_repair_requires_local_same_component_distance(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_locality_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 10),
                (10, 11),
                (11, 12),
                (12, 3),
                (0, 13),
                (3, 14),
                (0, 1),
                (1, 3),
            ],
            vertex_count=15,
        )

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(0, 10), (10, 11), (11, 12), (12, 3), (0, 13), (3, 14)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertFalse(mesh.edges[6].use_seam)
        self.assertFalse(mesh.edges[7].use_seam)
        report = [
            item for item in result.blender_two_edge_repair_candidate_reports
            if item['path_vertex_ids'] == [0, 1, 3]
        ][0]
        self.assertEqual(report['existing_seam_distance_between_endpoints'], 4)
        self.assertEqual(report['rejection_reason'], 'seam_distance_too_large')

    def test_two_edge_repair_safety_cap_prevents_mass_marking(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_cap_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path_edge_indices = build_many_two_edge_repair_candidates(17)

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertEqual(result.blender_two_edge_repair_allowed_candidates_total, 17)
        self.assertEqual(result.blender_two_edge_repair_safety_cap, 16)
        self.assertTrue(result.blender_two_edge_repair_over_cap)
        self.assertEqual(result.blender_two_edge_repair_paths_marked, 0)
        for first_index, second_index in path_edge_indices:
            self.assertFalse(mesh.edges[first_index].use_seam)
            self.assertFalse(mesh.edges[second_index].use_seam)

    def test_two_edge_repair_over_cap_marks_only_allowed_targets(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_target_cap_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path_edge_indices = build_many_two_edge_repair_candidates(
            15,
            include_targets=True,
        )
        target_indices = path_edge_indices[:2]
        non_target_indices = path_edge_indices[2:]

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertTrue(result.blender_two_edge_repair_over_cap)
        self.assertTrue(result.target_path_2045_2541_4884_marked)
        self.assertTrue(result.target_path_2045_2541_4884_accepted_by_target_over_cap_exception)
        self.assertTrue(result.target_path_2540_2541_2544_marked)
        self.assertTrue(result.target_path_2540_2541_2544_accepted_by_target_over_cap_exception)
        for first_index, second_index in target_indices:
            self.assertTrue(mesh.edges[first_index].use_seam)
            self.assertTrue(mesh.edges[second_index].use_seam)
        for first_index, second_index in non_target_indices:
            self.assertFalse(mesh.edges[first_index].use_seam)
            self.assertFalse(mesh.edges[second_index].use_seam)

    def test_two_edge_repair_target_a_telemetry(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_target_a_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (2045, 2540),
                (2540, 4884),
                (2045, 2046),
                (4884, 4885),
                (2045, 2541),
                (2541, 4884),
            ],
            vertex_count=4886,
        )

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(2045, 2540), (2540, 4884), (2045, 2046), (4884, 4885)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertTrue(result.target_path_2045_2541_4884_found)
        self.assertTrue(result.target_path_2045_2541_4884_allowed)
        self.assertTrue(result.target_path_2045_2541_4884_marked)
        self.assertTrue(result.target_path_2045_2541_4884_accepted_by_normal_rule)
        self.assertIsNone(result.target_path_2045_2541_4884_rejection_reason)

    def test_two_edge_repair_target_b_telemetry(self):
        seam_mapping = load_module('uvsp_seam_mapping_two_edge_target_b_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (2539, 2540),
                (2540, 4884),
                (2542, 2544),
                (2544, 4884),
                (2540, 2541),
                (2541, 2544),
            ],
            vertex_count=4885,
        )

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(2539, 2540), (2540, 4884), (2542, 2544), (2544, 4884)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertTrue(result.target_path_2540_2541_2544_found)
        self.assertTrue(result.target_path_2540_2541_2544_allowed)
        self.assertTrue(result.target_path_2540_2541_2544_marked)
        self.assertTrue(result.target_path_2540_2541_2544_accepted_by_normal_rule)
        self.assertIsNone(result.target_path_2540_2541_2544_rejection_reason)

    def test_two_edge_endpoint_bridge_marks_valid_inter_component_path(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_valid_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path_indices = build_endpoint_bridge_mesh()

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertTrue(mesh.edges[path_indices[0]].use_seam)
        self.assertTrue(mesh.edges[path_indices[1]].use_seam)
        self.assertEqual(result.blender_two_edge_endpoint_bridge_paths_marked, 1)
        self.assertEqual(result.blender_two_edge_endpoint_bridge_edges_marked, 2)
        report = [
            item for item in result.blender_two_edge_endpoint_bridge_candidate_reports
            if item['path_vertex_ids'] == [100, 101, 102]
        ][0]
        self.assertEqual(report['degree_pattern'], (1, 0, 1))
        self.assertFalse(report['accepted_by_target_over_cap_exception'])
        self.assertTrue(report['accepted_by_normal_rule'])

    def test_two_edge_endpoint_bridge_rejects_same_component_path(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_same_component_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path_indices = build_endpoint_bridge_mesh(same_component=True)

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertFalse(mesh.edges[path_indices[0]].use_seam)
        self.assertFalse(mesh.edges[path_indices[1]].use_seam)
        report = [
            item for item in result.blender_two_edge_endpoint_bridge_candidate_reports
            if item['path_vertex_ids'] == [100, 101, 102]
        ][0]
        self.assertEqual(report['rejection_reason'], 'same_component_not_endpoint_bridge')

    def test_two_edge_endpoint_bridge_rejects_degree_mismatches(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_degree_smoke', ADDON_DIR / 'seam_mapping.py')
        cases = (
            ({'endpoint_degrees': (2, 1)}, 'endpoint_not_degree_1'),
            ({'endpoint_degrees': (1, 2)}, 'endpoint_not_degree_1'),
            ({'intermediate_degree': 1}, 'intermediate_not_degree_0'),
        )
        for kwargs, reason in cases:
            with self.subTest(reason=reason, kwargs=kwargs):
                mesh, predicted_keys, path_indices = build_endpoint_bridge_mesh(**kwargs)

                result = seam_mapping.apply_seam_keys(
                    mesh,
                    predicted_keys,
                    clear_existing=True,
                    enable_local_repair=True,
                )

                self.assertFalse(mesh.edges[path_indices[0]].use_seam)
                self.assertFalse(mesh.edges[path_indices[1]].use_seam)
                report = [
                    item for item in result.blender_two_edge_endpoint_bridge_candidate_reports
                    if item['path_vertex_ids'] == [100, 101, 102]
                ][0]
                self.assertEqual(report['rejection_reason'], reason)

    def test_two_edge_endpoint_bridge_rejects_existing_or_missing_path_edges(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_edges_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path_indices = build_endpoint_bridge_mesh(first_gap_already_seam=True)

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertTrue(mesh.edges[path_indices[0]].use_seam)
        self.assertFalse(mesh.edges[path_indices[1]].use_seam)
        report = [
            item for item in result.blender_two_edge_endpoint_bridge_candidate_reports
            if item['path_vertex_ids'] == [100, 101, 102]
        ][0]
        self.assertEqual(report['rejection_reason'], 'edge_already_seam')

        missing = seam_mapping.apply_two_edge_endpoint_bridge_repair(
            FakeMesh(
                edges=[(90, 100), (102, 110), (100, 101)],
                vertex_count=10000,
                coords=endpoint_bridge_coords(),
            ),
            enabled=True,
            target_paths=((100, 101, 102),),
        )
        missing_report = missing['candidate_reports'][-1]
        self.assertEqual(missing_report['path_vertex_ids'], [100, 101, 102])
        self.assertEqual(missing_report['rejection_reason'], 'edge_not_found')

    def test_two_edge_endpoint_bridge_rejects_tangent_unavailable(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_no_tangent_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path_indices = build_endpoint_bridge_mesh(coords={})

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertFalse(mesh.edges[path_indices[0]].use_seam)
        report = [
            item for item in result.blender_two_edge_endpoint_bridge_candidate_reports
            if item['path_vertex_ids'] == [100, 101, 102]
        ][0]
        self.assertEqual(report['rejection_reason'], 'tangent_unavailable')

    def test_two_edge_endpoint_bridge_rejects_bad_tangent_alignment(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_alignment_smoke', ADDON_DIR / 'seam_mapping.py')
        coords = endpoint_bridge_coords()
        coords[90] = (0.02, 0.0, 0.0)
        mesh, predicted_keys, path_indices = build_endpoint_bridge_mesh(coords=coords)

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertFalse(mesh.edges[path_indices[0]].use_seam)
        report = [
            item for item in result.blender_two_edge_endpoint_bridge_candidate_reports
            if item['path_vertex_ids'] == [100, 101, 102]
        ][0]
        self.assertEqual(report['rejection_reason'], 'tangent_alignment_failed')
        self.assertLess(report['endpoint_tangent_alignment_u'], 0.0)

    def test_two_edge_endpoint_bridge_rejects_path_backtracking(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_backtrack_smoke', ADDON_DIR / 'seam_mapping.py')
        coords = endpoint_bridge_coords()
        coords[100] = (0.01, 0.0, 0.0)
        coords[101] = (0.02, 0.0, 0.0)
        coords[102] = (0.015, 0.0, 0.0)
        coords[110] = (0.014, 0.0, 0.0)
        mesh, predicted_keys, path_indices = build_endpoint_bridge_mesh(coords=coords)

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertFalse(mesh.edges[path_indices[0]].use_seam)
        report = [
            item for item in result.blender_two_edge_endpoint_bridge_candidate_reports
            if item['path_vertex_ids'] == [100, 101, 102]
        ][0]
        self.assertEqual(report['rejection_reason'], 'path_backtracking')
        self.assertLess(report['path_straightness'], -0.25)

    def test_two_edge_endpoint_bridge_rejects_path_too_long(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_length_smoke', ADDON_DIR / 'seam_mapping.py')
        coords = {
            90: (0.0, 0.0, 0.0),
            100: (10.0, 0.0, 0.0),
            101: (20.0, 0.0, 0.0),
            102: (30.0, 0.0, 0.0),
            110: (40.0, 0.0, 0.0),
        }
        mesh, predicted_keys, path_indices = build_endpoint_bridge_mesh(coords=coords)

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertFalse(mesh.edges[path_indices[0]].use_seam)
        report = [
            item for item in result.blender_two_edge_endpoint_bridge_candidate_reports
            if item['path_vertex_ids'] == [100, 101, 102]
        ][0]
        self.assertEqual(report['rejection_reason'], 'path_too_long')

    def test_two_edge_endpoint_bridge_snapshot_allows_shared_intermediate_targets(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_snapshot_smoke', ADDON_DIR / 'seam_mapping.py')
        coords = {
            2044: (-0.02, 0.0, 0.0),
            2045: (-0.01, 0.0, 0.0),
            2541: (0.0, 0.0, 0.0),
            4884: (0.01, 0.0, 0.0),
            4885: (0.02, 0.0, 0.0),
            2539: (0.0, -0.02, 0.0),
            2540: (0.0, -0.01, 0.0),
            2544: (0.0, 0.01, 0.0),
            2545: (0.0, 0.02, 0.0),
            9999: (1.0, 1.0, 1.0),
        }
        mesh = FakeMesh(
            edges=[
                (2044, 2045),
                (4884, 4885),
                (2539, 2540),
                (2544, 2545),
                (2045, 2541),
                (2541, 4884),
                (2540, 2541),
                (2541, 2544),
            ],
            vertex_count=10000,
            coords=coords,
        )

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(2044, 2045), (4884, 4885), (2539, 2540), (2544, 2545)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertTrue(result.target_path_2045_2541_4884_marked)
        self.assertTrue(result.target_path_2540_2541_2544_marked)
        self.assertGreaterEqual(result.blender_two_edge_endpoint_bridge_paths_marked, 2)
        self.assertGreaterEqual(result.blender_two_edge_endpoint_bridge_edges_marked, 4)

    def test_two_edge_endpoint_bridge_safety_cap_prevents_mass_marking(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_cap_smoke', ADDON_DIR / 'seam_mapping.py')
        edges = []
        predicted_keys = []
        coords = {9999: (1.0, 1.0, 1.0)}
        path_indices = []
        for index in range(9):
            base = 500 + index * 10
            edges.extend([(base - 1, base), (base + 2, base + 3)])
            predicted_keys.extend([(base - 1, base), (base + 2, base + 3)])
            path_indices.append((len(edges), len(edges) + 1))
            edges.extend([(base, base + 1), (base + 1, base + 2)])
            coords.update(endpoint_bridge_coords((base, base + 1, base + 2), base - 1, base + 3))
        mesh = FakeMesh(edges=edges, vertex_count=10000, coords=coords)

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertEqual(result.blender_two_edge_endpoint_bridge_allowed_total, 9)
        self.assertTrue(result.blender_two_edge_endpoint_bridge_over_cap)
        self.assertEqual(result.blender_two_edge_endpoint_bridge_paths_marked, 0)
        for first_index, second_index in path_indices:
            self.assertFalse(mesh.edges[first_index].use_seam)
            self.assertFalse(mesh.edges[second_index].use_seam)

    def test_two_edge_endpoint_bridge_over_cap_marks_only_allowed_targets(self):
        seam_mapping = load_module('uvsp_seam_mapping_endpoint_bridge_target_cap_smoke', ADDON_DIR / 'seam_mapping.py')
        coords = {
            2044: (-0.02, 0.0, 0.0),
            2045: (-0.01, 0.0, 0.0),
            2541: (0.0, 0.0, 0.0),
            4884: (0.01, 0.0, 0.0),
            4885: (0.02, 0.0, 0.0),
            2539: (0.0, -0.02, 0.0),
            2540: (0.0, -0.01, 0.0),
            2544: (0.0, 0.01, 0.0),
            2545: (0.0, 0.02, 0.0),
            9999: (1.0, 1.0, 1.0),
        }
        edges = [
            (2044, 2045),
            (4884, 4885),
            (2539, 2540),
            (2544, 2545),
            (2045, 2541),
            (2541, 4884),
            (2540, 2541),
            (2541, 2544),
        ]
        predicted_keys = [(2044, 2045), (4884, 4885), (2539, 2540), (2544, 2545)]
        non_target_indices = []
        for index in range(9):
            base = 600 + index * 10
            edges.extend([(base - 1, base), (base + 2, base + 3)])
            predicted_keys.extend([(base - 1, base), (base + 2, base + 3)])
            non_target_indices.append((len(edges), len(edges) + 1))
            edges.extend([(base, base + 1), (base + 1, base + 2)])
            coords.update(endpoint_bridge_coords((base, base + 1, base + 2), base - 1, base + 3))
        mesh = FakeMesh(edges=edges, vertex_count=10000, coords=coords)

        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertTrue(result.blender_two_edge_endpoint_bridge_over_cap)
        self.assertTrue(result.target_path_2045_2541_4884_marked)
        self.assertTrue(result.target_path_2045_2541_4884_accepted_by_target_over_cap_exception)
        self.assertTrue(result.target_path_2540_2541_2544_marked)
        self.assertTrue(result.target_path_2540_2541_2544_accepted_by_target_over_cap_exception)
        self.assertTrue(mesh.edges[4].use_seam)
        self.assertTrue(mesh.edges[5].use_seam)
        self.assertTrue(mesh.edges[6].use_seam)
        self.assertTrue(mesh.edges[7].use_seam)
        for first_index, second_index in non_target_indices:
            self.assertFalse(mesh.edges[first_index].use_seam)
            self.assertFalse(mesh.edges[second_index].use_seam)

    def test_local_repair_summary_reports_telemetry(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_summary_smoke', ADDON_DIR / 'seam_mapping.py')
        result = seam_mapping.SeamApplyResult(
            requested=1,
            unique=1,
            applied=1,
            ignored_non_original=0,
            duplicates_skipped=0,
            blender_local_repair_enabled=True,
            blender_local_repair_edges_marked=1,
            blender_local_repair_edges_rejected=2,
            blender_local_repair_allowed_candidates_total=1,
            blender_local_repair_repair_over_cap=False,
            human_case_2557_2558_found=True,
            human_case_2557_2558_marked_seam=True,
            human_case_2557_2558_degree_pattern=(2, 3),
            blender_two_edge_repair_paths_marked=2,
            blender_two_edge_repair_edges_marked=4,
            blender_two_edge_repair_allowed_candidates_total=2,
            blender_two_edge_repair_over_cap=False,
            target_path_2045_2541_4884_found=True,
            target_path_2045_2541_4884_marked=True,
            target_path_2045_2541_4884_tangent_alignments=(0.42, 0.36),
            target_path_2045_2541_4884_straightness=0.71,
            target_path_2540_2541_2544_found=True,
            target_path_2540_2541_2544_marked=True,
            target_path_2540_2541_2544_tangent_alignments=(0.31, 0.28),
            target_path_2540_2541_2544_straightness=0.62,
            blender_two_edge_endpoint_bridge_paths_marked=2,
            blender_two_edge_endpoint_bridge_edges_marked=4,
            blender_two_edge_endpoint_bridge_allowed_total=2,
            blender_two_edge_endpoint_bridge_over_cap=False,
        )

        summary = seam_mapping.format_apply_summary(result)

        self.assertIn('Local repair: 1 marked, 2 rejected, allowed=1, over_cap=false.', summary)
        self.assertIn('Human case [2557,2558]: marked, degree=(2, 3).', summary)
        self.assertIn('Two-edge repair: 2 paths marked, 4 edges marked, allowed=2, over_cap=false.', summary)
        self.assertIn(
            'Two-edge endpoint bridge: 2 paths marked, 4 edges marked, allowed=2, over_cap=false.',
            summary,
        )
        self.assertIn(
            'Target [2045,2541,4884]: marked, alignments=(0.42, 0.36), straightness=0.71.',
            summary,
        )
        self.assertIn(
            'Target [2540,2541,2544]: marked, alignments=(0.31, 0.28), straightness=0.62.',
            summary,
        )

    def test_topology_change_guard_blocks_stale_application(self):
        validation = load_module('uvsp_validation_smoke', ADDON_DIR / 'validation.py')
        obj = FakeObject(FakeMesh(edges=[(0, 1), (1, 2)], vertex_count=3))

        validation.require_unchanged_topology(obj, (3, 2))
        with self.assertRaisesRegex(ValueError, 'Mesh topology changed'):
            validation.require_unchanged_topology(obj, (3, 1))

    def test_weights_path_is_scene_level_not_preferences_level(self):
        prefs_source = read_addon_file('prefs.py')
        properties_source = read_addon_file('properties.py')
        ui_source = read_addon_file('ui.py')
        operators_source = read_addon_file('operators.py')
        validation_source = read_addon_file('validation.py')

        self.assertNotIn('model_weights_path', prefs_source)
        self.assertIn('model_weights_path', properties_source)
        self.assertIn('settings.model_weights_path', ui_source)
        self.assertIn('settings.model_weights_path', operators_source)
        self.assertIn('settings.model_weights_path', validation_source)

    def test_export_helper_does_not_require_pre_triangulated_mesh(self):
        export_source = read_addon_file('export_obj.py')
        operators_source = read_addon_file('operators.py')
        validation_source = read_addon_file('validation.py')

        self.assertIn('bmesh.new()', export_source)
        self.assertIn('bm.copy()', export_source)
        self.assertIn('bmesh.ops.triangulate', export_source)
        self.assertIn("quad_method='FIXED'", export_source)
        self.assertIn("ngon_method='BEAUTY'", export_source)
        self.assertNotIn('require_triangulated_mesh', operators_source)
        self.assertNotIn('require_triangulated_mesh', validation_source)

    def test_summary_wording_mentions_ignored_triangulation_only_edges(self):
        seam_mapping = load_module('uvsp_seam_mapping_summary_smoke', ADDON_DIR / 'seam_mapping.py')
        result = seam_mapping.SeamApplyResult(
            requested=5,
            unique=4,
            applied=3,
            ignored_non_original=1,
            duplicates_skipped=1,
        )

        summary = seam_mapping.format_apply_summary(result)

        self.assertIn('Marked 3 seam edges.', summary)
        self.assertIn('Ignored 1 triangulation-only edges.', summary)
        self.assertIn('Skipped 1 duplicates.', summary)

    def test_addon_does_not_branch_on_model_architecture(self):
        addon_text = '\n'.join(
            read_addon_file(name)
            for name in (
                '__init__.py',
                'prefs.py',
                'properties.py',
                'ui.py',
                'operators.py',
                'export_obj.py',
                'inference.py',
                'seam_mapping.py',
                'validation.py',
            )
        ).lower()

        self.assertNotIn('gatv2', addon_text)
        self.assertNotIn('graphsage', addon_text)
        self.assertNotIn('sparsemeshcnn', addon_text)
        self.assertNotIn('model-type', addon_text)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
