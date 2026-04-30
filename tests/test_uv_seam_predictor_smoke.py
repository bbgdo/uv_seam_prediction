import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


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
    def __init__(self, edges, vertex_count=None, coords=None, polygons=None):
        self.edges = [FakeEdge(edge, index) for index, edge in enumerate(edges)]
        coords = coords or {}
        if vertex_count is None:
            vertex_count = max(
                max((vertex for edge in edges for vertex in edge), default=-1),
                max(coords, default=-1),
            ) + 1
        self.vertices = [FakeVertex(coords.get(index)) for index in range(vertex_count)]
        self.polygons = [
            SimpleNamespace(vertices=tuple(polygon))
            for polygon in (polygons or [])
        ]
        self.update_count = 0

    def update(self):
        self.update_count += 1


class FakeObject:
    def __init__(self, mesh):
        self.name = 'FakeObject'
        self.type = 'MESH'
        self.data = mesh
        self.mode = 'OBJECT'

    def select_set(self, selected):
        self.selected = bool(selected)


class FakeModeSet:
    def __init__(self, bpy_module):
        self.bpy_module = bpy_module
        self.calls = []

    def __call__(self, *, mode):
        self.calls.append(mode)
        obj = self.bpy_module.context.object
        if obj is not None:
            obj.mode = mode
        return {'FINISHED'}

    def poll(self):
        return True


def load_operators_module_with_fakes(name, *, active_obj):
    package_name = f'{name}_pkg'
    package = ModuleType(package_name)
    package.__path__ = [str(ADDON_DIR)]
    sys.modules[package_name] = package

    bpy_module = ModuleType('bpy')
    bpy_module.types = SimpleNamespace(Operator=object)
    mode_set = FakeModeSet(bpy_module)
    bpy_module.ops = SimpleNamespace(object=SimpleNamespace(mode_set=mode_set))
    bpy_module.data = SimpleNamespace(objects={})
    bpy_module.context = SimpleNamespace(
        object=active_obj,
        view_layer=SimpleNamespace(objects=SimpleNamespace(active=active_obj)),
    )
    if active_obj is not None:
        bpy_module.data.objects[active_obj.name] = active_obj
    sys.modules['bpy'] = bpy_module

    calls = {
        'gap_fill': [],
        'dangling_cleanup': [],
        'seam_mirror': [],
        'inference': [],
        'export': [],
        'mode_set': mode_set.calls,
    }

    seam_mapping_module = ModuleType(f'{package_name}.seam_mapping')

    def fake_gap_fill(mesh, **kwargs):
        calls['gap_fill'].append((mesh, kwargs))
        return {
            'accepted_paths_count': 2,
            'accepted_edges_count': 3,
            'max_gap_hops': int(kwargs['max_gap_hops']),
        }

    def fake_dangling_cleanup(mesh, **kwargs):
        calls['dangling_cleanup'].append((mesh, kwargs))
        return {
            'removed_branches_count': 2,
            'removed_edges_count': 3,
            'max_dangling_edges': int(kwargs['max_dangling_edges']),
        }

    def fake_seam_mirror(mesh, **kwargs):
        calls['seam_mirror'].append((mesh, kwargs))
        return {
            'mirrored_edges_added': 2,
            'mirrored_edges_already_present': 1,
            'source_seam_edges': 5,
            'unmatched_vertices': 4,
            'missing_mirrored_edges': 1,
            'skipped_center_edges': 0,
            'direction': kwargs['direction'],
            'axis': kwargs['axis'],
            'tolerance': float(kwargs['tolerance']),
        }

    seam_mapping_module.apply_editable_shortest_path_gap_fill = fake_gap_fill
    seam_mapping_module.apply_editable_dangling_seam_cleanup = fake_dangling_cleanup
    seam_mapping_module.apply_editable_seam_mirror = fake_seam_mirror
    sys.modules[f'{package_name}.seam_mapping'] = seam_mapping_module

    validation_module = ModuleType(f'{package_name}.validation')

    def require_active_mesh_object(context):
        obj = context.view_layer.objects.active
        if obj is None:
            raise ValueError('Select an active mesh object.')
        if getattr(obj, 'type', None) != 'MESH':
            raise ValueError('Active object must be a mesh.')
        if getattr(obj, 'data', None) is None:
            raise ValueError('Active mesh object has no mesh data.')
        return obj

    validation_module.require_active_mesh_object = require_active_mesh_object
    validation_module.bpy_path_to_os_path = lambda path: path
    sys.modules[f'{package_name}.validation'] = validation_module

    inference_module = ModuleType(f'{package_name}.inference')

    def fail_inference(*args, **kwargs):
        calls['inference'].append((args, kwargs))
        raise AssertionError('manual gap fill must not run inference')

    inference_module.create_temp_work_files = fail_inference
    inference_module.launch_inference = fail_inference
    inference_module.has_timed_out = fail_inference
    inference_module.poll_job = fail_inference
    inference_module.close_log_handles = fail_inference
    inference_module.read_text_tail = fail_inference
    inference_module.terminate_job = fail_inference
    inference_module.cleanup_job = fail_inference
    sys.modules[f'{package_name}.inference'] = inference_module

    export_module = ModuleType(f'{package_name}.export_obj')

    def fail_export(*args, **kwargs):
        calls['export'].append((args, kwargs))
        raise AssertionError('manual gap fill must not export OBJ')

    export_module.export_object_to_obj_with_hidden_triangulation = fail_export
    sys.modules[f'{package_name}.export_obj'] = export_module

    module = load_module(f'{package_name}.operators', ADDON_DIR / 'operators.py')
    return module, calls


class UVSeamPredictorSmokeTests(unittest.TestCase):
    def test_editable_gap_fill_fills_one_hop_gap(self):
        seam_mapping = load_module('uvsp_editable_gap_one_hop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (2, 3), (1, 2)], vertex_count=4)
        mesh.edges[0].use_seam = True
        mesh.edges[1].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['accepted_edges_count'], 1)
        self.assertTrue(mesh.edges[2].use_seam)
        self.assertEqual(result['accepted_paths'][0]['vertices'], [1, 2])

    def test_editable_gap_fill_fills_two_hop_gap(self):
        seam_mapping = load_module('uvsp_editable_gap_two_hop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (3, 4), (1, 2), (2, 3)], vertex_count=5)
        mesh.edges[0].use_seam = True
        mesh.edges[1].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['accepted_edges_count'], 2)
        self.assertTrue(mesh.edges[2].use_seam)
        self.assertTrue(mesh.edges[3].use_seam)
        self.assertEqual(result['accepted_paths'][0]['vertices'], [1, 2, 3])

    def test_editable_gap_fill_three_hop_gap_requires_matching_limit(self):
        seam_mapping = load_module('uvsp_editable_gap_three_hop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (4, 5), (1, 2), (2, 3), (3, 4)], vertex_count=6)
        mesh.edges[0].use_seam = True
        mesh.edges[1].use_seam = True

        blocked = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)
        self.assertEqual(blocked['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[2].use_seam)
        self.assertFalse(mesh.edges[3].use_seam)
        self.assertFalse(mesh.edges[4].use_seam)

        filled = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=3)
        self.assertEqual(filled['accepted_paths_count'], 1)
        self.assertEqual(filled['accepted_edges_count'], 3)
        self.assertTrue(mesh.edges[2].use_seam)
        self.assertTrue(mesh.edges[3].use_seam)
        self.assertTrue(mesh.edges[4].use_seam)

    def test_editable_gap_fill_attaches_endpoint_to_existing_seam_vertex_one_hop(self):
        seam_mapping = load_module('uvsp_editable_gap_existing_vertex_one_hop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (3, 4), (3, 5), (1, 3)], vertex_count=6)
        before_edge_count = len(mesh.edges)
        for index in range(3):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(len(mesh.edges), before_edge_count)
        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['endpoint_to_existing_seam_candidates'], 1)
        self.assertEqual(result['endpoint_to_existing_seam_accepted'], 1)
        self.assertTrue(mesh.edges[3].use_seam)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'endpoint_to_existing_seam_vertex')
        self.assertEqual(result['accepted_paths'][0]['vertices'], [1, 3])

    def test_editable_gap_fill_attaches_endpoint_to_existing_seam_vertex_two_hops(self):
        seam_mapping = load_module('uvsp_editable_gap_existing_vertex_two_hop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (4, 5), (4, 6), (1, 2), (2, 4)], vertex_count=7)
        for index in range(3):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['accepted_edges_count'], 2)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'endpoint_to_existing_seam_vertex')
        self.assertEqual(result['accepted_paths'][0]['vertices'], [1, 2, 4])
        self.assertTrue(mesh.edges[3].use_seam)
        self.assertTrue(mesh.edges[4].use_seam)

    def test_editable_gap_fill_existing_seam_vertex_three_hop_respects_limit(self):
        seam_mapping = load_module('uvsp_editable_gap_existing_vertex_three_hop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (5, 6), (5, 7), (1, 2), (2, 3), (3, 5)], vertex_count=8)
        for index in range(3):
            mesh.edges[index].use_seam = True

        blocked = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)
        self.assertEqual(blocked['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[3].use_seam)
        self.assertFalse(mesh.edges[4].use_seam)
        self.assertFalse(mesh.edges[5].use_seam)

        filled = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=3)
        self.assertEqual(filled['accepted_paths_count'], 1)
        self.assertEqual(filled['accepted_paths'][0]['kind'], 'endpoint_to_existing_seam_vertex')
        self.assertEqual(filled['accepted_paths'][0]['vertices'], [1, 2, 3, 5])
        self.assertTrue(mesh.edges[3].use_seam)
        self.assertTrue(mesh.edges[4].use_seam)
        self.assertTrue(mesh.edges[5].use_seam)

    def test_editable_gap_fill_marks_only_existing_edges(self):
        seam_mapping = load_module('uvsp_editable_gap_existing_edges_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (3, 4), (1, 2), (2, 3)], vertex_count=5)
        original_keys = {tuple(sorted(edge.vertices)) for edge in mesh.edges}
        mesh.edges[0].use_seam = True
        mesh.edges[1].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)
        marked_keys = {tuple(sorted(edge.vertices)) for edge in mesh.edges if edge.use_seam}

        self.assertEqual(result['accepted_edges_count'], 2)
        self.assertTrue(marked_keys <= original_keys)
        self.assertEqual(len(mesh.edges), 4)

    def test_editable_gap_fill_keeps_perfect_existing_seam_unchanged(self):
        seam_mapping = load_module('uvsp_editable_gap_perfect_seam_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2)], vertex_count=3)
        for edge in mesh.edges:
            edge.use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertTrue(all(edge.use_seam for edge in mesh.edges))

    def test_editable_gap_fill_rejects_same_component_by_default(self):
        seam_mapping = load_module('uvsp_editable_gap_same_component_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (2, 3), (0, 3)], vertex_count=4)
        mesh.edges[0].use_seam = True
        mesh.edges[1].use_seam = True
        mesh.edges[2].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[3].use_seam)
        self.assertGreaterEqual(result['rejected_same_component'], 1)

    def test_editable_gap_fill_rejects_same_component_existing_seam_target_by_default(self):
        seam_mapping = load_module('uvsp_editable_gap_existing_vertex_same_component_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (0, 3), (3, 4), (1, 3)], vertex_count=5)
        for index in range(3):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[3].use_seam)
        self.assertGreaterEqual(result['rejected_endpoint_to_existing_same_component'], 1)

    def test_editable_gap_fill_closes_one_edge_same_component_endpoint_loop_gap(self):
        seam_mapping = load_module('uvsp_editable_gap_endpoint_loop_one_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)],
            vertex_count=7,
        )
        for index in range(6):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['endpoint_loop_closure_candidates'], 1)
        self.assertEqual(result['endpoint_loop_closure_accepted'], 1)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'endpoint_to_endpoint')
        self.assertTrue(result['accepted_paths'][0]['same_component_loop_closure'])
        self.assertEqual(result['accepted_paths'][0]['vertices'], [0, 6])
        self.assertTrue(mesh.edges[6].use_seam)

    def test_editable_gap_fill_closes_two_edge_same_component_endpoint_loop_gap(self):
        seam_mapping = load_module('uvsp_editable_gap_endpoint_loop_two_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                (0, 7), (7, 6),
            ],
            vertex_count=8,
        )
        for index in range(6):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['endpoint_loop_closure_accepted'], 1)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'endpoint_to_endpoint')
        self.assertTrue(result['accepted_paths'][0]['same_component_loop_closure'])
        self.assertEqual(result['accepted_paths'][0]['vertices'], [0, 7, 6])
        self.assertTrue(mesh.edges[6].use_seam)
        self.assertTrue(mesh.edges[7].use_seam)

    def test_editable_gap_fill_closes_same_component_endpoint_to_degree_two_vertex(self):
        seam_mapping = load_module('uvsp_editable_gap_endpoint_loop_degree_two_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                (0, 6),
            ],
            vertex_count=8,
        )
        for index in range(7):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['endpoint_to_existing_loop_closure_candidates'], 1)
        self.assertEqual(result['endpoint_to_existing_loop_closure_accepted'], 1)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'endpoint_to_existing_seam_vertex')
        self.assertTrue(result['accepted_paths'][0]['same_component_loop_closure'])
        self.assertEqual(result['accepted_paths'][0]['vertices'], [0, 6])
        self.assertTrue(mesh.edges[7].use_seam)

    def test_editable_gap_fill_closes_same_component_endpoint_to_junction(self):
        seam_mapping = load_module('uvsp_editable_gap_endpoint_loop_junction_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                (6, 7), (6, 8), (0, 6),
            ],
            vertex_count=9,
        )
        for index in range(8):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['endpoint_to_existing_loop_closure_accepted'], 1)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'endpoint_to_existing_seam_vertex')
        self.assertTrue(result['accepted_paths'][0]['same_component_loop_closure'])
        self.assertEqual(result['accepted_paths'][0]['vertices'], [0, 6])
        self.assertTrue(mesh.edges[8].use_seam)

    def test_editable_gap_fill_same_component_endpoint_loop_three_hops_respects_limit(self):
        seam_mapping = load_module('uvsp_editable_gap_endpoint_loop_three_hop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 6),
            ],
            vertex_count=9,
        )
        for index in range(6):
            mesh.edges[index].use_seam = True

        blocked = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)
        self.assertEqual(blocked['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[6].use_seam)
        self.assertFalse(mesh.edges[7].use_seam)
        self.assertFalse(mesh.edges[8].use_seam)

        filled = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=3)
        self.assertEqual(filled['accepted_paths_count'], 1)
        self.assertEqual(filled['endpoint_loop_closure_accepted'], 1)
        self.assertEqual(filled['accepted_paths'][0]['kind'], 'endpoint_to_endpoint')
        self.assertTrue(filled['accepted_paths'][0]['same_component_loop_closure'])
        self.assertEqual(filled['accepted_paths'][0]['vertices'], [0, 7, 8, 6])
        self.assertTrue(mesh.edges[6].use_seam)
        self.assertTrue(mesh.edges[7].use_seam)
        self.assertTrue(mesh.edges[8].use_seam)

    def test_editable_gap_fill_rejects_tiny_same_component_endpoint_shortcut(self):
        seam_mapping = load_module('uvsp_editable_gap_endpoint_loop_tiny_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (2, 3), (0, 3)], vertex_count=4)
        for index in range(3):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[3].use_seam)
        self.assertGreaterEqual(result['rejected_endpoint_same_component_too_short'], 1)

    def test_editable_gap_fill_rejects_tiny_same_component_endpoint_to_degree_two_shortcut(self):
        seam_mapping = load_module('uvsp_editable_gap_endpoint_loop_tiny_degree_two_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (2, 3), (0, 2)], vertex_count=4)
        for index in range(3):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[3].use_seam)
        self.assertGreaterEqual(result['rejected_endpoint_to_existing_same_component_too_short'], 1)

    def test_editable_gap_fill_same_component_endpoint_closure_rejects_internal_seam_vertex(self):
        seam_mapping = load_module('uvsp_editable_gap_endpoint_loop_internal_seam_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                (0, 3), (3, 6),
            ],
            vertex_count=7,
        )
        for index in range(6):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[6].use_seam)
        self.assertFalse(mesh.edges[7].use_seam)
        self.assertGreaterEqual(result['rejected_internal_seam_vertex'], 1)

    def test_editable_gap_fill_same_component_endpoint_closure_rejects_existing_seam_edge(self):
        seam_mapping = load_module('uvsp_editable_gap_endpoint_loop_existing_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                (0, 3),
            ],
            vertex_count=7,
        )
        for index in range(6):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=3)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[6].use_seam)
        self.assertGreaterEqual(result['rejected_existing_seam_internal'], 1)

    def test_editable_gap_fill_rejects_internal_existing_seam_vertex(self):
        seam_mapping = load_module('uvsp_editable_gap_internal_seam_vertex_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1),
                (0, 2),
                (2, 8),
                (5, 6),
                (5, 7),
                (1, 2),
                (2, 3),
                (3, 5),
            ],
            vertex_count=10,
        )
        for index in range(5):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=3)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertGreaterEqual(result['rejected_internal_seam_vertex'], 1)
        self.assertFalse(mesh.edges[5].use_seam)
        self.assertFalse(mesh.edges[6].use_seam)
        self.assertFalse(mesh.edges[7].use_seam)

    def test_editable_gap_fill_rejects_path_with_existing_seam_edge(self):
        seam_mapping = load_module('uvsp_editable_gap_existing_seam_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1),
                (4, 5),
                (2, 3),
                (2, 6),
                (3, 7),
                (1, 2),
                (3, 4),
            ],
            vertex_count=8,
        )
        before_edge_count = len(mesh.edges)
        for index in range(5):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=3)

        self.assertEqual(len(mesh.edges), before_edge_count)
        self.assertEqual(result['accepted_paths_count'], 2)
        self.assertGreaterEqual(result['rejected_existing_seam_internal'], 1)
        self.assertTrue(mesh.edges[5].use_seam)
        self.assertTrue(mesh.edges[6].use_seam)

    def test_editable_gap_fill_deterministic_when_multiple_candidates_exist(self):
        seam_mapping = load_module('uvsp_editable_gap_deterministic_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (4, 5), (6, 7), (9, 10), (1, 4), (7, 9)], vertex_count=11)
        for index in range(4):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 2)
        self.assertEqual([path['vertices'] for path in result['accepted_paths']], [[1, 4], [7, 9]])

    def test_editable_gap_fill_skips_second_candidate_after_endpoint_consumed(self):
        seam_mapping = load_module('uvsp_editable_gap_consumed_endpoint_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1),
                (3, 4),
                (5, 6),
                (1, 2),
                (2, 3),
                (1, 7),
                (7, 5),
            ],
            vertex_count=8,
        )
        for index in range(3):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['accepted_paths'][0]['vertices'], [1, 2, 3])
        self.assertEqual(result['accepted_edges_count'], 2)
        self.assertGreaterEqual(result['rejected_conflict_consumed_endpoint'], 1)
        self.assertTrue(mesh.edges[3].use_seam)
        self.assertTrue(mesh.edges[4].use_seam)
        self.assertFalse(mesh.edges[5].use_seam)
        self.assertFalse(mesh.edges[6].use_seam)

    def test_editable_gap_fill_skips_candidate_reusing_reserved_bridge_edge(self):
        seam_mapping = load_module('uvsp_editable_gap_reserved_edge_conflict_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1),
                (5, 11),
                (6, 12),
                (10, 13),
                (1, 2),
                (2, 3),
                (3, 5),
                (6, 7),
                (7, 2),
                (3, 8),
                (8, 10),
            ],
            vertex_count=14,
        )
        for index in range(4):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=5)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['accepted_paths'][0]['vertices'], [1, 2, 3, 5])
        self.assertGreaterEqual(result['rejected_conflict_consumed_endpoint'], 1)
        self.assertTrue(mesh.edges[4].use_seam)
        self.assertTrue(mesh.edges[5].use_seam)
        self.assertTrue(mesh.edges[6].use_seam)
        self.assertFalse(mesh.edges[7].use_seam)
        self.assertFalse(mesh.edges[8].use_seam)
        self.assertFalse(mesh.edges[9].use_seam)
        self.assertFalse(mesh.edges[10].use_seam)

    def test_editable_gap_fill_existing_seam_target_is_not_consumed(self):
        seam_mapping = load_module('uvsp_editable_gap_existing_vertex_unconsumed_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (2, 3), (5, 6), (5, 7), (1, 5), (3, 5)], vertex_count=8)
        for index in range(4):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 2)
        self.assertEqual(result['endpoint_to_existing_seam_accepted'], 2)
        self.assertEqual(
            [path['vertices'] for path in result['accepted_paths']],
            [[1, 5], [3, 5]],
        )
        self.assertTrue(mesh.edges[4].use_seam)
        self.assertTrue(mesh.edges[5].use_seam)

    def test_editable_gap_fill_prefers_shorter_existing_seam_target_over_longer_endpoint(self):
        seam_mapping = load_module('uvsp_editable_gap_existing_vertex_order_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (3, 4), (10, 11), (10, 12), (1, 2), (2, 3), (1, 10)],
            vertex_count=13,
        )
        for index in range(4):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'endpoint_to_existing_seam_vertex')
        self.assertEqual(result['accepted_paths'][0]['vertices'], [1, 10])
        self.assertTrue(mesh.edges[6].use_seam)
        self.assertFalse(mesh.edges[4].use_seam)
        self.assertFalse(mesh.edges[5].use_seam)

    def test_editable_gap_fill_fills_one_hop_junction_gap(self):
        seam_mapping = load_module('uvsp_editable_gap_junction_one_hop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (0, 2), (0, 3), (4, 5), (4, 6), (0, 4)],
            vertex_count=7,
        )
        for index in range(5):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['junction_gap_candidates'], 1)
        self.assertEqual(result['junction_gap_accepted'], 1)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'junction_gap_closure')
        self.assertEqual(result['accepted_paths'][0]['vertices'], [0, 4])
        self.assertTrue(mesh.edges[5].use_seam)

    def test_editable_gap_fill_fills_two_hop_junction_gap_when_allowed(self):
        seam_mapping = load_module('uvsp_editable_gap_junction_two_hop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (0, 2), (0, 3), (5, 6), (5, 7), (0, 4), (4, 5)],
            vertex_count=8,
        )
        for index in range(5):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['junction_gap_accepted'], 1)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'junction_gap_closure')
        self.assertEqual(result['accepted_paths'][0]['vertices'], [0, 4, 5])
        self.assertTrue(mesh.edges[5].use_seam)
        self.assertTrue(mesh.edges[6].use_seam)

    def test_editable_gap_fill_three_hop_junction_gap_respects_limit(self):
        seam_mapping = load_module('uvsp_editable_gap_junction_three_hop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (0, 2), (0, 3), (6, 7), (6, 8), (0, 4), (4, 5), (5, 6)],
            vertex_count=9,
        )
        for index in range(5):
            mesh.edges[index].use_seam = True

        blocked = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)
        self.assertEqual(blocked['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[5].use_seam)
        self.assertFalse(mesh.edges[6].use_seam)
        self.assertFalse(mesh.edges[7].use_seam)

        filled = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=3)
        self.assertEqual(filled['accepted_paths_count'], 1)
        self.assertEqual(filled['accepted_paths'][0]['kind'], 'junction_gap_closure')
        self.assertEqual(filled['accepted_paths'][0]['vertices'], [0, 4, 5, 6])
        self.assertTrue(mesh.edges[5].use_seam)
        self.assertTrue(mesh.edges[6].use_seam)
        self.assertTrue(mesh.edges[7].use_seam)

    def test_editable_gap_fill_junction_gap_requires_high_degree_endpoint(self):
        seam_mapping = load_module('uvsp_editable_gap_junction_high_degree_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (0, 2), (3, 4), (3, 5), (0, 3)], vertex_count=6)
        for index in range(4):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[4].use_seam)
        self.assertGreaterEqual(result['rejected_junction_gap_no_high_degree_endpoint'], 1)

    def test_editable_gap_fill_junction_gap_rejects_internal_seam_vertex(self):
        seam_mapping = load_module('uvsp_editable_gap_junction_internal_seam_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (0, 2), (0, 8), (0, 11), (11, 3), (3, 9),
                (5, 6), (5, 7), (0, 3), (3, 5),
            ],
            vertex_count=12,
        )
        for index in range(8):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[8].use_seam)
        self.assertFalse(mesh.edges[9].use_seam)
        self.assertGreaterEqual(result['rejected_junction_gap_internal_seam_vertex'], 1)

    def test_editable_gap_fill_junction_gap_rejects_existing_seam_edge_in_path(self):
        seam_mapping = load_module('uvsp_editable_gap_junction_existing_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (0, 2), (0, 8), (0, 11), (11, 3), (3, 9),
                (3, 4), (4, 5), (0, 3),
            ],
            vertex_count=12,
        )
        for index in range(8):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[8].use_seam)
        self.assertGreaterEqual(result['rejected_junction_gap_existing_seam_edge'], 1)

    def test_editable_gap_fill_junction_same_component_loop_guard_accepts_large_loop(self):
        seam_mapping = load_module('uvsp_editable_gap_junction_large_loop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                (0, 7), (0, 8), (6, 9), (0, 6),
            ],
            vertex_count=10,
        )
        for index in range(9):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'junction_gap_closure')
        self.assertEqual(result['accepted_paths'][0]['vertices'], [0, 6])
        self.assertTrue(mesh.edges[9].use_seam)

    def test_editable_gap_fill_junction_same_component_loop_guard_rejects_tiny_chord(self):
        seam_mapping = load_module('uvsp_editable_gap_junction_tiny_chord_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (1, 2), (2, 3), (0, 7), (0, 8), (3, 9), (0, 3)],
            vertex_count=10,
        )
        for index in range(6):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 0)
        self.assertFalse(mesh.edges[6].use_seam)
        self.assertGreaterEqual(result['rejected_junction_gap_same_component_too_short'], 1)

    def test_editable_gap_fill_junction_gap_does_not_consume_junction_endpoints(self):
        seam_mapping = load_module('uvsp_editable_gap_junction_unconsumed_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (0, 2), (0, 3),
                (4, 5), (4, 6),
                (7, 8), (7, 9),
                (0, 4), (0, 7),
            ],
            vertex_count=10,
        )
        for index in range(7):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=1)

        self.assertEqual(result['accepted_paths_count'], 2)
        self.assertEqual(result['junction_gap_accepted'], 2)
        self.assertEqual(
            [path['vertices'] for path in result['accepted_paths']],
            [[0, 4], [0, 7]],
        )
        self.assertTrue(mesh.edges[7].use_seam)
        self.assertTrue(mesh.edges[8].use_seam)

    def test_editable_gap_fill_junction_gap_rejects_reserved_bridge_edge_conflict(self):
        seam_mapping = load_module('uvsp_editable_gap_junction_reserved_conflict_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (0, 1), (0, 2), (0, 3),
                (4, 5), (4, 6),
                (7, 8), (7, 9),
                (0, 10), (10, 4), (10, 7),
            ],
            vertex_count=11,
        )
        for index in range(7):
            mesh.edges[index].use_seam = True

        result = seam_mapping.apply_editable_shortest_path_gap_fill(mesh, max_gap_hops=2)

        self.assertEqual(result['accepted_paths_count'], 1)
        self.assertEqual(result['accepted_paths'][0]['kind'], 'junction_gap_closure')
        self.assertEqual(result['accepted_paths'][0]['vertices'], [0, 10, 4])
        self.assertGreaterEqual(result['rejected_junction_gap_reserved_edge'], 1)
        self.assertTrue(mesh.edges[7].use_seam)
        self.assertTrue(mesh.edges[8].use_seam)
        self.assertFalse(mesh.edges[9].use_seam)

    def test_dangling_cleanup_removes_one_edge_endpoint_to_junction_spur(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_one_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)], vertex_count=8)
        for edge in mesh.edges:
            edge.use_seam = True

        result = seam_mapping.apply_editable_dangling_seam_cleanup(
            mesh,
            max_dangling_edges=1,
            protect_boundary_vertices=False,
        )

        self.assertEqual(result['removed_branches_count'], 1)
        self.assertEqual(result['removed_edges_count'], 1)
        self.assertFalse(mesh.edges[0].use_seam)
        self.assertTrue(all(edge.use_seam for edge in mesh.edges[1:]))
        self.assertEqual(result['removed_branches'][0]['start_vertex'], 0)
        self.assertEqual(result['removed_branches'][0]['terminal_vertex'], 1)
        self.assertEqual(result['removed_branches'][0]['length'], 1)

    def test_dangling_cleanup_removes_two_edge_spur_only_when_allowed(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_two_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        edges = [(0, 8), (8, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]
        blocked_mesh = FakeMesh(edges=edges, vertex_count=9)
        for edge in blocked_mesh.edges:
            edge.use_seam = True

        blocked = seam_mapping.apply_editable_dangling_seam_cleanup(
            blocked_mesh,
            max_dangling_edges=1,
            protect_boundary_vertices=False,
        )

        self.assertEqual(blocked['removed_branches_count'], 0)
        self.assertGreaterEqual(blocked['rejected_too_long'], 1)
        self.assertTrue(all(edge.use_seam for edge in blocked_mesh.edges))

        filled_mesh = FakeMesh(edges=edges, vertex_count=9)
        for edge in filled_mesh.edges:
            edge.use_seam = True
        filled = seam_mapping.apply_editable_dangling_seam_cleanup(
            filled_mesh,
            max_dangling_edges=2,
            protect_boundary_vertices=False,
        )

        self.assertEqual(filled['removed_branches_count'], 1)
        self.assertEqual(filled['removed_edges_count'], 2)
        self.assertFalse(filled_mesh.edges[0].use_seam)
        self.assertFalse(filled_mesh.edges[1].use_seam)
        self.assertTrue(all(edge.use_seam for edge in filled_mesh.edges[2:]))

    def test_dangling_cleanup_does_not_remove_long_branch(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_long_branch_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 8), (8, 9), (9, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)], vertex_count=10)
        for edge in mesh.edges:
            edge.use_seam = True

        result = seam_mapping.apply_editable_dangling_seam_cleanup(
            mesh,
            max_dangling_edges=2,
            protect_boundary_vertices=False,
        )

        self.assertEqual(result['removed_branches_count'], 0)
        self.assertGreaterEqual(result['rejected_too_long'], 1)
        self.assertTrue(all(edge.use_seam for edge in mesh.edges))

    def test_dangling_cleanup_removes_one_edge_isolated_seam_component(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_isolated_one_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (2, 3)], vertex_count=4)
        mesh.edges[0].use_seam = True

        result = seam_mapping.apply_editable_dangling_seam_cleanup(
            mesh,
            max_dangling_edges=3,
            protect_boundary_vertices=False,
        )

        self.assertEqual(result['removed_branches_count'], 1)
        self.assertEqual(result['removed_edges_count'], 1)
        self.assertEqual(result['isolated_path_candidates'], 1)
        self.assertEqual(result['isolated_paths_removed'], 1)
        self.assertEqual(result['isolated_path_edges_removed'], 1)
        self.assertFalse(mesh.edges[0].use_seam)
        self.assertFalse(mesh.edges[1].use_seam)
        self.assertEqual(result['removed_branches'][0]['kind'], 'isolated_path_component')

    def test_dangling_cleanup_removes_two_edge_isolated_path_only_when_allowed(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_isolated_two_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        edges = [(0, 1), (1, 2)]
        blocked_mesh = FakeMesh(edges=edges, vertex_count=3)
        for edge in blocked_mesh.edges:
            edge.use_seam = True

        blocked = seam_mapping.apply_editable_dangling_seam_cleanup(
            blocked_mesh,
            max_dangling_edges=1,
            protect_boundary_vertices=False,
        )

        self.assertEqual(blocked['isolated_paths_removed'], 0)
        self.assertGreaterEqual(blocked['rejected_isolated_path_too_long'], 1)
        self.assertTrue(all(edge.use_seam for edge in blocked_mesh.edges))

        removed_mesh = FakeMesh(edges=edges, vertex_count=3)
        for edge in removed_mesh.edges:
            edge.use_seam = True

        removed = seam_mapping.apply_editable_dangling_seam_cleanup(
            removed_mesh,
            max_dangling_edges=2,
            protect_boundary_vertices=False,
        )

        self.assertEqual(removed['removed_branches_count'], 1)
        self.assertEqual(removed['isolated_paths_removed'], 1)
        self.assertEqual(removed['isolated_path_edges_removed'], 2)
        self.assertTrue(all(not edge.use_seam for edge in removed_mesh.edges))

    def test_dangling_cleanup_does_not_remove_long_isolated_path(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_isolated_long_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (2, 3)], vertex_count=4)
        for edge in mesh.edges:
            edge.use_seam = True

        result = seam_mapping.apply_editable_dangling_seam_cleanup(
            mesh,
            max_dangling_edges=2,
            protect_boundary_vertices=False,
        )

        self.assertEqual(result['removed_branches_count'], 0)
        self.assertGreaterEqual(result['rejected_isolated_path_too_long'], 1)
        self.assertTrue(all(edge.use_seam for edge in mesh.edges))

    def test_dangling_cleanup_does_not_remove_closed_loop(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_loop_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (0, 2)], vertex_count=3)
        for edge in mesh.edges:
            edge.use_seam = True

        result = seam_mapping.apply_editable_dangling_seam_cleanup(
            mesh,
            max_dangling_edges=3,
            protect_boundary_vertices=False,
        )

        self.assertEqual(result['candidates_total'], 0)
        self.assertEqual(result['removed_branches_count'], 0)
        self.assertGreaterEqual(result['rejected_isolated_path_not_simple'], 1)
        self.assertTrue(all(edge.use_seam for edge in mesh.edges))

    def test_dangling_cleanup_does_not_remove_isolated_component_with_junction(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_isolated_junction_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)], vertex_count=4)
        for edge in mesh.edges:
            edge.use_seam = True

        result = seam_mapping.apply_editable_dangling_seam_cleanup(
            mesh,
            max_dangling_edges=5,
            protect_boundary_vertices=False,
        )

        self.assertEqual(result['removed_branches_count'], 0)
        self.assertGreaterEqual(result['rejected_isolated_path_not_simple'], 1)
        self.assertTrue(all(edge.use_seam for edge in mesh.edges))

    def test_dangling_cleanup_boundary_protects_isolated_path_when_enabled(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_isolated_boundary_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1)],
            vertex_count=3,
            polygons=[(0, 1, 2)],
        )
        mesh.edges[0].use_seam = True

        protected = seam_mapping.apply_editable_dangling_seam_cleanup(
            mesh,
            max_dangling_edges=1,
            protect_boundary_vertices=True,
        )

        self.assertEqual(protected['removed_branches_count'], 0)
        self.assertGreaterEqual(protected['rejected_isolated_path_boundary_protected'], 1)
        self.assertTrue(mesh.edges[0].use_seam)

        allowed_mesh = FakeMesh(
            edges=[(0, 1)],
            vertex_count=3,
            polygons=[(0, 1, 2)],
        )
        allowed_mesh.edges[0].use_seam = True

        allowed = seam_mapping.apply_editable_dangling_seam_cleanup(
            allowed_mesh,
            max_dangling_edges=1,
            protect_boundary_vertices=False,
        )

        self.assertEqual(allowed['isolated_paths_removed'], 1)
        self.assertFalse(allowed_mesh.edges[0].use_seam)

    def test_dangling_cleanup_protects_boundary_endpoint_when_enabled(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_boundary_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)],
            vertex_count=9,
            polygons=[(0, 1, 8)],
        )
        for edge in mesh.edges:
            edge.use_seam = True

        result = seam_mapping.apply_editable_dangling_seam_cleanup(
            mesh,
            max_dangling_edges=1,
            protect_boundary_vertices=True,
        )

        self.assertEqual(result['removed_branches_count'], 0)
        self.assertGreaterEqual(result['rejected_boundary_protected'], 1)
        self.assertTrue(mesh.edges[0].use_seam)

    def test_dangling_cleanup_allows_non_boundary_spur_cleanup(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_non_boundary_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)], vertex_count=8)
        for edge in mesh.edges:
            edge.use_seam = True

        result = seam_mapping.apply_editable_dangling_seam_cleanup(
            mesh,
            max_dangling_edges=1,
            protect_boundary_vertices=True,
        )

        self.assertEqual(result['removed_branches_count'], 1)
        self.assertFalse(mesh.edges[0].use_seam)

    def test_dangling_cleanup_removes_multiple_branches_deterministically(self):
        seam_mapping = load_module('uvsp_dangling_cleanup_deterministic_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(3, 10), (1, 10), (4, 10), (2, 10)], vertex_count=11)
        for edge in mesh.edges:
            edge.use_seam = True

        result = seam_mapping.apply_editable_dangling_seam_cleanup(
            mesh,
            max_dangling_edges=1,
            protect_boundary_vertices=False,
        )

        self.assertEqual(
            [branch['start_vertex'] for branch in result['removed_branches']],
            [1, 2, 3],
        )
        self.assertEqual(result['removed_branches_count'], 3)
        self.assertGreaterEqual(result['rejected_entire_component'], 1)

    def test_seam_mirror_axis_x_negative_to_positive_mirrors_correctly(self):
        seam_mapping = load_module('uvsp_seam_mirror_x_neg_to_pos_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (2, 3)],
            coords={
                0: (-1.0, 0.0, 0.0),
                1: (-1.0, 1.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (1.0, 1.0, 0.0),
            },
        )
        before_edge_count = len(mesh.edges)
        mesh.edges[0].use_seam = True

        result = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='NEGATIVE_TO_POSITIVE',
            axis='X',
            tolerance=1e-4,
        )

        self.assertEqual(len(mesh.edges), before_edge_count)
        self.assertTrue(mesh.edges[0].use_seam)
        self.assertTrue(mesh.edges[1].use_seam)
        self.assertEqual(result['source_seam_edges'], 1)
        self.assertEqual(result['mirrored_edges_added'], 1)
        self.assertEqual(result['mirrored_edges'][0]['mirrored_edge_key'], [2, 3])

    def test_seam_mirror_axis_x_positive_to_negative_mirrors_correctly(self):
        seam_mapping = load_module('uvsp_seam_mirror_x_pos_to_neg_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (2, 3)],
            coords={
                0: (-1.0, 0.0, 0.0),
                1: (-1.0, 1.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (1.0, 1.0, 0.0),
            },
        )
        mesh.edges[1].use_seam = True

        result = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='POSITIVE_TO_NEGATIVE',
            axis='X',
            tolerance=1e-4,
        )

        self.assertTrue(mesh.edges[0].use_seam)
        self.assertTrue(mesh.edges[1].use_seam)
        self.assertEqual(result['mirrored_edges_added'], 1)
        self.assertEqual(result['mirrored_edges'][0]['mirrored_edge_key'], [0, 1])

    def test_seam_mirror_axis_y_negative_to_positive_mirrors_correctly(self):
        seam_mapping = load_module('uvsp_seam_mirror_y_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (2, 3)],
            coords={
                0: (0.0, -1.0, 0.0),
                1: (1.0, -1.0, 0.0),
                2: (0.0, 1.0, 0.0),
                3: (1.0, 1.0, 0.0),
            },
        )
        mesh.edges[0].use_seam = True

        result = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='NEGATIVE_TO_POSITIVE',
            axis='Y',
            tolerance=1e-4,
        )

        self.assertTrue(mesh.edges[1].use_seam)
        self.assertEqual(result['axis'], 'Y')
        self.assertEqual(result['mirrored_edges_added'], 1)

    def test_seam_mirror_axis_z_negative_to_positive_mirrors_correctly(self):
        seam_mapping = load_module('uvsp_seam_mirror_z_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (2, 3)],
            coords={
                0: (0.0, 0.0, -1.0),
                1: (1.0, 0.0, -1.0),
                2: (0.0, 0.0, 1.0),
                3: (1.0, 0.0, 1.0),
            },
        )
        mesh.edges[0].use_seam = True

        result = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='NEGATIVE_TO_POSITIVE',
            axis='Z',
            tolerance=1e-4,
        )

        self.assertTrue(mesh.edges[1].use_seam)
        self.assertEqual(result['axis'], 'Z')
        self.assertEqual(result['mirrored_edges_added'], 1)

    def test_seam_mirror_is_additive_when_destination_already_seam(self):
        seam_mapping = load_module('uvsp_seam_mirror_additive_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (2, 3)],
            coords={
                0: (-1.0, 0.0, 0.0),
                1: (-1.0, 1.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (1.0, 1.0, 0.0),
            },
        )
        for edge in mesh.edges:
            edge.use_seam = True

        result = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='NEGATIVE_TO_POSITIVE',
            axis='X',
            tolerance=1e-4,
        )

        self.assertTrue(all(edge.use_seam for edge in mesh.edges))
        self.assertEqual(result['mirrored_edges_added'], 0)
        self.assertEqual(result['mirrored_edges_already_present'], 1)

    def test_seam_mirror_skips_center_plane_edges(self):
        seam_mapping = load_module('uvsp_seam_mirror_center_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1)],
            coords={
                0: (0.0, 0.0, 0.0),
                1: (0.0, 1.0, 0.0),
            },
        )
        mesh.edges[0].use_seam = True

        result = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='NEGATIVE_TO_POSITIVE',
            axis='X',
            tolerance=1e-4,
        )

        self.assertTrue(mesh.edges[0].use_seam)
        self.assertEqual(result['source_seam_edges'], 0)
        self.assertEqual(result['mirrored_edges_added'], 0)
        self.assertEqual(result['skipped_center_edges'], 1)

    def test_seam_mirror_skips_unmatched_vertices(self):
        seam_mapping = load_module('uvsp_seam_mirror_unmatched_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (2, 3)],
            vertex_count=4,
            coords={
                0: (-1.0, 0.0, 0.0),
                1: (-1.0, 1.0, 0.0),
                2: (1.0, 0.0, 0.0),
            },
        )
        mesh.edges[0].use_seam = True

        result = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='NEGATIVE_TO_POSITIVE',
            axis='X',
            tolerance=1e-4,
        )

        self.assertTrue(mesh.edges[0].use_seam)
        self.assertFalse(mesh.edges[1].use_seam)
        self.assertEqual(result['mirrored_edges_added'], 0)
        self.assertEqual(result['unmatched_vertices'], 1)

    def test_seam_mirror_skips_missing_mirrored_edge(self):
        seam_mapping = load_module('uvsp_seam_mirror_missing_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1)],
            coords={
                0: (-1.0, 0.0, 0.0),
                1: (-1.0, 1.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (1.0, 1.0, 0.0),
            },
        )
        before_edge_count = len(mesh.edges)
        mesh.edges[0].use_seam = True

        result = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='NEGATIVE_TO_POSITIVE',
            axis='X',
            tolerance=1e-4,
        )

        self.assertEqual(len(mesh.edges), before_edge_count)
        self.assertTrue(mesh.edges[0].use_seam)
        self.assertEqual(result['mirrored_edges_added'], 0)
        self.assertEqual(result['missing_mirrored_edges'], 1)

    def test_seam_mirror_tolerance_controls_vertex_matching(self):
        seam_mapping = load_module('uvsp_seam_mirror_tolerance_smoke', ADDON_DIR / 'seam_mapping.py')
        within = FakeMesh(
            edges=[(0, 1), (2, 3)],
            coords={
                0: (-1.0, 0.0, 0.0),
                1: (-1.0, 1.0, 0.0),
                2: (1.00005, 0.0, 0.0),
                3: (1.00005, 1.0, 0.0),
            },
        )
        within.edges[0].use_seam = True
        within_result = seam_mapping.apply_editable_seam_mirror(
            within,
            direction='NEGATIVE_TO_POSITIVE',
            axis='X',
            tolerance=1e-4,
        )

        beyond = FakeMesh(
            edges=[(0, 1), (2, 3)],
            coords={
                0: (-1.0, 0.0, 0.0),
                1: (-1.0, 1.0, 0.0),
                2: (1.001, 0.0, 0.0),
                3: (1.001, 1.0, 0.0),
            },
        )
        beyond.edges[0].use_seam = True
        beyond_result = seam_mapping.apply_editable_seam_mirror(
            beyond,
            direction='NEGATIVE_TO_POSITIVE',
            axis='X',
            tolerance=1e-4,
        )

        self.assertEqual(within_result['mirrored_edges_added'], 1)
        self.assertTrue(within.edges[1].use_seam)
        self.assertEqual(beyond_result['mirrored_edges_added'], 0)
        self.assertEqual(beyond_result['unmatched_vertices'], 1)
        self.assertFalse(beyond.edges[1].use_seam)

    def test_seam_mirror_direction_filter_uses_source_side_only(self):
        seam_mapping = load_module('uvsp_seam_mirror_direction_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (2, 3)],
            coords={
                0: (-1.0, 0.0, 0.0),
                1: (-1.0, 1.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (1.0, 1.0, 0.0),
            },
        )
        mesh.edges[1].use_seam = True

        negative_to_positive = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='NEGATIVE_TO_POSITIVE',
            axis='X',
            tolerance=1e-4,
        )
        self.assertEqual(negative_to_positive['source_seam_edges'], 0)
        self.assertFalse(mesh.edges[0].use_seam)

        positive_to_negative = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='POSITIVE_TO_NEGATIVE',
            axis='X',
            tolerance=1e-4,
        )

        self.assertEqual(positive_to_negative['source_seam_edges'], 1)
        self.assertEqual(positive_to_negative['mirrored_edges_added'], 1)
        self.assertTrue(mesh.edges[0].use_seam)

    def test_seam_mirror_no_source_seams_reports_source_zero(self):
        seam_mapping = load_module('uvsp_seam_mirror_no_source_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (2, 3)],
            coords={
                0: (-1.0, 0.0, 0.0),
                1: (-1.0, 1.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (1.0, 1.0, 0.0),
            },
        )

        result = seam_mapping.apply_editable_seam_mirror(
            mesh,
            direction='NEGATIVE_TO_POSITIVE',
            axis='X',
            tolerance=1e-4,
        )

        self.assertEqual(result['source_seam_edges'], 0)
        self.assertEqual(result['mirrored_edges_added'], 0)
        self.assertFalse(any(edge.use_seam for edge in mesh.edges))

    def test_apply_seam_keys_uses_editable_gap_fill(self):
        seam_mapping = load_module('uvsp_editable_gap_routing_smoke', ADDON_DIR / 'seam_mapping.py')

        mesh = FakeMesh(edges=[(0, 1), (3, 4), (1, 2), (2, 3)], vertex_count=5)
        result = seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1), (3, 4)],
            clear_existing=True,
            enable_local_repair=True,
            fill_small_gaps=True,
            fill_gap_max_hops=2,
        )

        self.assertEqual(result.editable_gap_fill_result['accepted_paths_count'], 1)
        self.assertTrue(mesh.edges[2].use_seam)
        self.assertTrue(mesh.edges[3].use_seam)

    def test_editable_gap_hops_property_uses_soft_recommended_max_only(self):
        properties_source = read_addon_file('properties.py')
        start = properties_source.index('postprocess_fill_gap_max_hops')
        end = properties_source.index('manual_cleanup_max_dangling_edges')
        gap_hops_source = properties_source[start:end]

        self.assertIn('default=2', gap_hops_source)
        self.assertIn('min=1', gap_hops_source)
        self.assertIn('soft_max=3', gap_hops_source)
        self.assertNotRegex(gap_hops_source, r'(?m)^\s*max=3,')
        self.assertIn('3 is recommended; higher values may over-connect seams.', gap_hops_source)

    def test_manual_dangling_cleanup_properties_are_defined(self):
        properties_source = read_addon_file('properties.py')

        self.assertIn('manual_cleanup_max_dangling_edges', properties_source)
        self.assertIn("name='Max Dangling Length'", properties_source)
        self.assertIn('default=1', properties_source)
        self.assertIn('soft_max=3', properties_source)
        self.assertIn('manual_cleanup_protect_boundary_vertices', properties_source)
        self.assertIn("name='Protect Boundary Ends'", properties_source)
        self.assertIn('Do not remove dangling branches anchored at mesh boundary vertices.', properties_source)

    def test_manual_mirror_axis_and_tolerance_properties_are_defined(self):
        properties_source = read_addon_file('properties.py')

        self.assertIn('manual_mirror_axis', properties_source)
        self.assertIn("name='Mirror Axis'", properties_source)
        self.assertIn("('X', 'X', 'Local X axis')", properties_source)
        self.assertIn("('Y', 'Y', 'Local Y axis')", properties_source)
        self.assertIn("('Z', 'Z', 'Local Z axis')", properties_source)
        self.assertIn("default='X'", properties_source)
        self.assertIn('Local mesh axis used for manual seam mirroring.', properties_source)
        self.assertIn('manual_mirror_tolerance', properties_source)
        self.assertIn("name='Mirror Tolerance'", properties_source)
        self.assertIn('default=1e-4', properties_source)
        self.assertIn('min=1e-8', properties_source)
        self.assertIn('soft_max=1e-2', properties_source)
        self.assertIn(
            'Coordinate tolerance used to match mirrored vertices in object-local space.',
            properties_source,
        )

    def test_manual_fill_current_seams_operator_calls_existing_gap_filler(self):
        mesh = FakeMesh(edges=[(0, 1), (2, 3), (1, 2)], vertex_count=4)
        obj = FakeObject(mesh)
        operators, calls = load_operators_module_with_fakes(
            'uvsp_manual_gap_fill_operator_smoke',
            active_obj=obj,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(
                uv_seam_predictor_settings=SimpleNamespace(
                    postprocess_fill_gap_max_hops=5,
                    last_run_summary='',
                )
            ),
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
        )
        reports = []
        operator = operators.UVSEAM_OT_fill_current_seam_gaps()
        operator.report = lambda levels, message: reports.append((levels, message))

        result = operator.execute(context)

        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(len(calls['gap_fill']), 1)
        called_mesh, kwargs = calls['gap_fill'][0]
        self.assertIs(called_mesh, mesh)
        self.assertEqual(kwargs, {
            'enabled': True,
            'max_gap_hops': 5,
            'allow_same_component': False,
        })
        self.assertEqual(calls['inference'], [])
        self.assertEqual(calls['export'], [])
        self.assertEqual(
            context.scene.uv_seam_predictor_settings.last_run_summary,
            'Filled 2 seam gap paths / 3 edges with max hops 5.',
        )
        self.assertEqual(reports[-1], ({'INFO'}, 'Filled 2 seam gap paths / 3 edges with max hops 5.'))

    def test_manual_fill_current_seams_operator_restores_edit_mode(self):
        mesh = FakeMesh(edges=[(0, 1), (2, 3), (1, 2)], vertex_count=4)
        obj = FakeObject(mesh)
        obj.mode = 'EDIT'
        operators, calls = load_operators_module_with_fakes(
            'uvsp_manual_gap_fill_edit_mode_smoke',
            active_obj=obj,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(
                uv_seam_predictor_settings=SimpleNamespace(
                    postprocess_fill_gap_max_hops=2,
                    last_run_summary='',
                )
            ),
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
        )
        operator = operators.UVSEAM_OT_fill_current_seam_gaps()
        operator.report = lambda levels, message: None

        result = operator.execute(context)

        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(calls['mode_set'], ['OBJECT', 'EDIT'])
        self.assertEqual(obj.mode, 'EDIT')

    def test_manual_fill_current_seams_operator_cancels_for_non_mesh(self):
        obj = SimpleNamespace(name='Camera', type='CAMERA', data=None, mode='OBJECT')
        operators, calls = load_operators_module_with_fakes(
            'uvsp_manual_gap_fill_non_mesh_smoke',
            active_obj=obj,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(
                uv_seam_predictor_settings=SimpleNamespace(
                    postprocess_fill_gap_max_hops=2,
                    last_run_summary='',
                )
            ),
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
        )
        reports = []
        operator = operators.UVSEAM_OT_fill_current_seam_gaps()
        operator.report = lambda levels, message: reports.append((levels, message))

        result = operator.execute(context)

        self.assertEqual(result, {'CANCELLED'})
        self.assertEqual(calls['gap_fill'], [])
        self.assertEqual(calls['inference'], [])
        self.assertEqual(calls['export'], [])
        self.assertEqual(
            context.scene.uv_seam_predictor_settings.last_run_summary,
            'Active object must be a mesh.',
        )
        self.assertEqual(reports[-1], ({'WARNING'}, 'Active object must be a mesh.'))

    def test_manual_dangling_cleanup_operator_calls_existing_cleanup_function(self):
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (1, 3)], vertex_count=4)
        obj = FakeObject(mesh)
        operators, calls = load_operators_module_with_fakes(
            'uvsp_manual_dangling_cleanup_operator_smoke',
            active_obj=obj,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(
                uv_seam_predictor_settings=SimpleNamespace(
                    manual_cleanup_max_dangling_edges=2,
                    manual_cleanup_protect_boundary_vertices=True,
                    last_run_summary='',
                )
            ),
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
        )
        reports = []
        operator = operators.UVSEAM_OT_clean_small_dangling_seams()
        operator.report = lambda levels, message: reports.append((levels, message))

        result = operator.execute(context)

        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(len(calls['dangling_cleanup']), 1)
        called_mesh, kwargs = calls['dangling_cleanup'][0]
        self.assertIs(called_mesh, mesh)
        self.assertEqual(kwargs, {
            'enabled': True,
            'max_dangling_edges': 2,
            'protect_boundary_vertices': True,
            'allow_remove_entire_component': False,
        })
        self.assertEqual(calls['gap_fill'], [])
        self.assertEqual(calls['inference'], [])
        self.assertEqual(calls['export'], [])
        self.assertEqual(
            context.scene.uv_seam_predictor_settings.last_run_summary,
            'Removed 2 dangling seam branches / 3 edges with max length 2.',
        )
        self.assertEqual(
            reports[-1],
            ({'INFO'}, 'Removed 2 dangling seam branches / 3 edges with max length 2.'),
        )

    def test_manual_dangling_cleanup_operator_restores_edit_mode(self):
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (1, 3)], vertex_count=4)
        obj = FakeObject(mesh)
        obj.mode = 'EDIT'
        operators, calls = load_operators_module_with_fakes(
            'uvsp_manual_dangling_cleanup_edit_mode_smoke',
            active_obj=obj,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(
                uv_seam_predictor_settings=SimpleNamespace(
                    manual_cleanup_max_dangling_edges=1,
                    manual_cleanup_protect_boundary_vertices=False,
                    last_run_summary='',
                )
            ),
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
        )
        operator = operators.UVSEAM_OT_clean_small_dangling_seams()
        operator.report = lambda levels, message: None

        result = operator.execute(context)

        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(calls['mode_set'], ['OBJECT', 'EDIT'])
        self.assertEqual(obj.mode, 'EDIT')

    def test_manual_dangling_cleanup_operator_cancels_for_non_mesh(self):
        obj = SimpleNamespace(name='Camera', type='CAMERA', data=None, mode='OBJECT')
        operators, calls = load_operators_module_with_fakes(
            'uvsp_manual_dangling_cleanup_non_mesh_smoke',
            active_obj=obj,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(
                uv_seam_predictor_settings=SimpleNamespace(
                    manual_cleanup_max_dangling_edges=1,
                    manual_cleanup_protect_boundary_vertices=True,
                    last_run_summary='',
                )
            ),
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
        )
        reports = []
        operator = operators.UVSEAM_OT_clean_small_dangling_seams()
        operator.report = lambda levels, message: reports.append((levels, message))

        result = operator.execute(context)

        self.assertEqual(result, {'CANCELLED'})
        self.assertEqual(calls['dangling_cleanup'], [])
        self.assertEqual(calls['inference'], [])
        self.assertEqual(calls['export'], [])
        self.assertEqual(
            context.scene.uv_seam_predictor_settings.last_run_summary,
            'Active object must be a mesh.',
        )
        self.assertEqual(reports[-1], ({'WARNING'}, 'Active object must be a mesh.'))

    def test_manual_seam_mirror_operators_call_helper_with_direction_and_tolerance(self):
        mesh = FakeMesh(edges=[(0, 1), (2, 3)], vertex_count=4)
        obj = FakeObject(mesh)
        operators, calls = load_operators_module_with_fakes(
            'uvsp_manual_seam_mirror_operator_smoke',
            active_obj=obj,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(
                uv_seam_predictor_settings=SimpleNamespace(
                    manual_mirror_axis='Y',
                    manual_mirror_tolerance=0.0025,
                    last_run_summary='',
                )
            ),
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
        )
        reports = []
        left_operator = operators.UVSEAM_OT_mirror_current_seams_left_to_right()
        right_operator = operators.UVSEAM_OT_mirror_current_seams_right_to_left()
        left_operator.report = lambda levels, message: reports.append((levels, message))
        right_operator.report = lambda levels, message: reports.append((levels, message))

        left_result = left_operator.execute(context)
        right_result = right_operator.execute(context)

        self.assertEqual(left_result, {'FINISHED'})
        self.assertEqual(right_result, {'FINISHED'})
        self.assertEqual(len(calls['seam_mirror']), 2)
        self.assertEqual(calls['seam_mirror'][0][1], {
            'enabled': True,
            'direction': 'NEGATIVE_TO_POSITIVE',
            'axis': 'Y',
            'tolerance': 0.0025,
            'skip_center_edges': True,
        })
        self.assertEqual(calls['seam_mirror'][1][1], {
            'enabled': True,
            'direction': 'POSITIVE_TO_NEGATIVE',
            'axis': 'Y',
            'tolerance': 0.0025,
            'skip_center_edges': True,
        })
        self.assertEqual(calls['gap_fill'], [])
        self.assertEqual(calls['dangling_cleanup'], [])
        self.assertEqual(calls['inference'], [])
        self.assertEqual(calls['export'], [])
        self.assertEqual(
            reports[0],
            (
                {'INFO'},
                'Mirror Y −→+: added 2, already 1, source 5, unmatched vertices 4, '
                'missing edges 1, skipped center 0.',
            ),
        )
        self.assertEqual(
            reports[1],
            (
                {'INFO'},
                'Mirror Y +→−: added 2, already 1, source 5, unmatched vertices 4, '
                'missing edges 1, skipped center 0.',
            ),
        )

    def test_manual_seam_mirror_operator_restores_edit_mode(self):
        mesh = FakeMesh(edges=[(0, 1), (2, 3)], vertex_count=4)
        obj = FakeObject(mesh)
        obj.mode = 'EDIT'
        operators, calls = load_operators_module_with_fakes(
            'uvsp_manual_seam_mirror_edit_mode_smoke',
            active_obj=obj,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(
                uv_seam_predictor_settings=SimpleNamespace(
                    manual_mirror_axis='X',
                    manual_mirror_tolerance=1e-4,
                    last_run_summary='',
                )
            ),
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
        )
        operator = operators.UVSEAM_OT_mirror_current_seams_left_to_right()
        operator.report = lambda levels, message: None

        result = operator.execute(context)

        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(calls['mode_set'], ['OBJECT', 'EDIT'])
        self.assertEqual(obj.mode, 'EDIT')

    def test_manual_seam_mirror_operator_cancels_for_non_mesh(self):
        obj = SimpleNamespace(name='Camera', type='CAMERA', data=None, mode='OBJECT')
        operators, calls = load_operators_module_with_fakes(
            'uvsp_manual_seam_mirror_non_mesh_smoke',
            active_obj=obj,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(
                uv_seam_predictor_settings=SimpleNamespace(
                    manual_mirror_axis='X',
                    manual_mirror_tolerance=1e-4,
                    last_run_summary='',
                )
            ),
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
        )
        reports = []
        operator = operators.UVSEAM_OT_mirror_current_seams_right_to_left()
        operator.report = lambda levels, message: reports.append((levels, message))

        result = operator.execute(context)

        self.assertEqual(result, {'CANCELLED'})
        self.assertEqual(calls['seam_mirror'], [])
        self.assertEqual(calls['inference'], [])
        self.assertEqual(calls['export'], [])
        self.assertEqual(
            context.scene.uv_seam_predictor_settings.last_run_summary,
            'Active object must be a mesh.',
        )
        self.assertEqual(reports[-1], ({'WARNING'}, 'Active object must be a mesh.'))

    def test_manual_fill_current_seams_button_and_registration_are_wired(self):
        operators_source = read_addon_file('operators.py')
        ui_source = read_addon_file('ui.py')
        init_source = read_addon_file('__init__.py')

        self.assertIn("bl_idname = 'uv_seam_predictor.fill_current_seam_gaps'", operators_source)
        self.assertIn("bl_label = 'Fill Gaps on Current Seams'", operators_source)
        self.assertIn(
            "bl_description = 'Fill small gaps in the currently marked seam edges using editable mesh topology'",
            operators_source,
        )
        self.assertIn("manual_box.label(text='Manual Seam Cleanup')", ui_source)
        self.assertIn("manual_box.operator('uv_seam_predictor.fill_current_seam_gaps'", ui_source)
        self.assertIn('operators.UVSEAM_OT_fill_current_seam_gaps', init_source)
        self.assertIn("bl_idname = 'uv_seam_predictor.clean_small_dangling_seams'", operators_source)
        self.assertIn("bl_label = 'Clean Small Dangling Seams'", operators_source)
        self.assertIn(
            "bl_description = 'Remove short dangling seam branches from the currently marked seams'",
            operators_source,
        )
        self.assertIn("manual_box.operator('uv_seam_predictor.clean_small_dangling_seams'", ui_source)
        self.assertIn("manual_box.prop(settings, 'manual_cleanup_max_dangling_edges')", ui_source)
        self.assertIn("manual_box.prop(settings, 'manual_cleanup_protect_boundary_vertices')", ui_source)
        self.assertIn('operators.UVSEAM_OT_clean_small_dangling_seams', init_source)
        self.assertIn("bl_idname = 'uv_seam_predictor.mirror_current_seams_l_to_r'", operators_source)
        self.assertIn("bl_idname = 'uv_seam_predictor.mirror_current_seams_r_to_l'", operators_source)
        self.assertIn("bl_label = 'Mirror Seams −→+'", operators_source)
        self.assertIn("bl_label = 'Mirror Seams +→−'", operators_source)
        self.assertIn(
            'Mirror current seam flags from the negative side of the selected local axis',
            operators_source,
        )
        self.assertIn(
            'Mirror current seam flags from the positive side of the selected local axis',
            operators_source,
        )
        self.assertIn("manual_box.prop(settings, 'manual_mirror_axis')", ui_source)
        self.assertIn("manual_box.prop(settings, 'manual_mirror_tolerance')", ui_source)
        self.assertIn("manual_box.operator('uv_seam_predictor.mirror_current_seams_l_to_r'", ui_source)
        self.assertIn("manual_box.operator('uv_seam_predictor.mirror_current_seams_r_to_l'", ui_source)
        self.assertIn('operators.UVSEAM_OT_mirror_current_seams_left_to_right', init_source)
        self.assertIn('operators.UVSEAM_OT_mirror_current_seams_right_to_left', init_source)

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
            postprocess_d_max=3,
            postprocess_r_bridge=6,
            postprocess_l_min=4,
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
            postprocess_d_max=3,
            postprocess_r_bridge=6,
            postprocess_l_min=4,
            postprocess_anchor_boundary=True,
        )
        args = inference.build_cli_args(prefs, settings, 'mesh.obj', 'out.json')
        # New flags present:
        for flag in (
            '--postprocess-tau-low',
            '--postprocess-d-max',
            '--postprocess-r-bridge',
            '--postprocess-l-min',
        ):
            self.assertIn(flag, args, f'expected new flag {flag} in cmd')
        self.assertIn('--postprocess-anchor-boundary', args)
        for suffix in ('tau-high', 'epsilon'):
            self.assertNotIn('--postprocess-' + suffix, args)
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
            postprocess_d_max=3,
            postprocess_r_bridge=6,
            postprocess_l_min=4,
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
        self.assertEqual(result.editable_gap_fill_result['accepted_edges_count'], 0)

    def test_summary_reports_core_counts_bridge_and_gap_fill(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_summary_smoke', ADDON_DIR / 'seam_mapping.py')

        result_bare = seam_mapping.SeamApplyResult(
            requested=5,
            unique=4,
            applied=3,
            ignored_non_original=1,
            duplicates_skipped=1,
        )
        summary_bare = seam_mapping.format_apply_summary(result_bare)
        self.assertIn('Marked 3 seam edges.', summary_bare)
        self.assertIn('Ignored 1 triangulation-only edges.', summary_bare)
        self.assertIn('Skipped 1 duplicates.', summary_bare)
        self.assertNotIn('Bridge:', summary_bare)
        self.assertNotIn('Gap fill:', summary_bare)

        result_bridge = seam_mapping.SeamApplyResult(
            requested=2,
            unique=2,
            applied=2,
            ignored_non_original=0,
            duplicates_skipped=0,
            accepted_bridge_edges_present_in_json=3,
            accepted_bridge_edges_applied=2,
            accepted_bridge_edges_ignored_non_original=1,
        )
        summary_bridge = seam_mapping.format_apply_summary(result_bridge)
        self.assertIn('Bridge: 3 accepted in JSON, 2 applied, 1 ignored as non-original.', summary_bridge)

        result_gap = seam_mapping.SeamApplyResult(
            requested=2,
            unique=2,
            applied=2,
            ignored_non_original=0,
            duplicates_skipped=0,
            editable_gap_fill_result={
                'accepted_paths_count': 2,
                'accepted_edges_count': 4,
                'max_gap_hops': 2,
            },
        )
        summary_gap = seam_mapping.format_apply_summary(result_gap)
        self.assertIn('Gap fill: 2 paths filled, 4 edges added.', summary_gap)

        result_no_gap = seam_mapping.SeamApplyResult(
            requested=1,
            unique=1,
            applied=1,
            ignored_non_original=0,
            duplicates_skipped=0,
            editable_gap_fill_result={'accepted_paths_count': 0, 'accepted_edges_count': 0},
        )
        self.assertNotIn('Gap fill:', seam_mapping.format_apply_summary(result_no_gap))

        for legacy_phrase in (
            'Local repair:',
            'Two-edge repair:',
            'Two-edge endpoint bridge:',
            'Curved two-edge endpoint bridge:',
            'Tangent-audit endpoint bridge:',
            'Human case [2557,2558]:',
            'Human Phase 2B.1',
            'Target [2045',
        ):
            self.assertNotIn(legacy_phrase, summary_bare)

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
