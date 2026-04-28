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


class FakeMesh:
    def __init__(self, edges, vertex_count=None):
        self.edges = [FakeEdge(edge, index) for index, edge in enumerate(edges)]
        if vertex_count is None:
            vertex_count = max((vertex for edge in edges for vertex in edge), default=-1) + 1
        self.vertices = [object() for _ in range(vertex_count)]
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
        )

        summary = seam_mapping.format_apply_summary(result)

        self.assertIn('Local repair: 1 marked, 2 rejected, allowed=1, over_cap=false.', summary)
        self.assertIn('Human case [2557,2558]: marked, degree=(2, 3).', summary)

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
