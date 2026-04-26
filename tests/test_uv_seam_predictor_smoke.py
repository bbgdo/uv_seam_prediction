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
    def __init__(self, vertices):
        self.vertices = vertices
        self.use_seam = False


class FakeMesh:
    def __init__(self, edges, vertex_count=None):
        self.edges = [FakeEdge(edge) for edge in edges]
        if vertex_count is None:
            vertex_count = max((vertex for edge in edges for vertex in edge), default=-1) + 1
        self.vertices = [object() for _ in range(vertex_count)]
        self.update_count = 0

    def update(self):
        self.update_count += 1


class FakeObject:
    def __init__(self, mesh):
        self.data = mesh


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
            postprocess_seam_threshold=0.5,
            postprocess_alpha_cost=0.5,
            postprocess_tau_bridge=0.28,
            postprocess_conf_floor=0.10,
            postprocess_max_low_conf_fraction=0.50,
            postprocess_force_close_max_edges=3,
            postprocess_r_self=8,
            postprocess_r_cross=10,
            postprocess_ambiguity_margin=0.05,
            postprocess_garbage_max_edges=4,
            postprocess_r_snap=3,
            postprocess_snap_max_edges=12,
            postprocess_r_band=2,
            postprocess_eta_main=0.35,
        )

        args = inference.build_cli_args(prefs, settings, 'mesh.obj', 'out.json')

        self.assertIn('--mesh-path', args)
        self.assertIn('--model-weights', args)
        self.assertIn('--threshold', args)
        self.assertIn('--output-json', args)
        self.assertNotIn('--feature-bundle', args)

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
