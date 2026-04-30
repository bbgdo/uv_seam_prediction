import importlib.util
import inspect
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


def append_endpoint_bridge_candidate(edges, predicted_keys, coords, base, total_span=0.02, y=0.0):
    left_neighbor = base - 1
    u = base
    middle = base + 1
    v = base + 2
    right_neighbor = base + 3
    step = total_span / 2.0
    coords[left_neighbor] = (-step, y, 0.0)
    coords[u] = (0.0, y, 0.0)
    coords[middle] = (step, y, 0.0)
    coords[v] = (total_span, y, 0.0)
    coords[right_neighbor] = (total_span + step, y, 0.0)
    edges.extend([(left_neighbor, u), (v, right_neighbor)])
    predicted_keys.extend([(left_neighbor, u), (v, right_neighbor)])
    path_edge_indices = (len(edges), len(edges) + 1)
    edges.extend([(u, middle), (middle, v)])
    return path_edge_indices, (u, middle, v)


def append_curved_endpoint_bridge_candidate(edges, predicted_keys, coords, base, y=0.0):
    left_neighbor = base - 1
    u = base
    middle = base + 1
    v = base + 2
    right_neighbor = base + 3
    coords[left_neighbor] = (-1.0, y, 0.0)
    coords[u] = (0.0, y, 0.0)
    coords[middle] = (1.0, y, 0.0)
    coords[v] = (1.2, y + 0.98, 0.0)
    coords[right_neighbor] = (1.4, y + 1.96, 0.0)
    edges.extend([(left_neighbor, u), (v, right_neighbor)])
    predicted_keys.extend([(left_neighbor, u), (v, right_neighbor)])
    path_edge_indices = (len(edges), len(edges) + 1)
    edges.extend([(u, middle), (middle, v)])
    return path_edge_indices, (u, middle, v)


def build_curved_endpoint_bridge_mesh(count=1, *, vertex_count=10000):
    edges = []
    predicted_keys = []
    coords = {9999: (1000.0, 1000.0, 1000.0)}
    paths = []
    for index in range(count):
        _, path = append_curved_endpoint_bridge_candidate(
            edges,
            predicted_keys,
            coords,
            100 + index * 10,
            y=0.0,
        )
        paths.append(path)
    return FakeMesh(edges=edges, vertex_count=vertex_count, coords=coords), predicted_keys, paths


def build_straight_tangent_weak_mesh(path=(100, 101, 102), *, alternative_support=False, same_component=False):
    u, middle, v = path
    left_neighbor = u - 1
    right_neighbor = v + 1
    alt_neighbor = u - 2
    edges = [(left_neighbor, u), (v, right_neighbor), (u, middle), (middle, v)]
    predicted_keys = [(left_neighbor, u), (v, right_neighbor)]
    if alternative_support:
        edges.append((alt_neighbor, u))
    if same_component:
        edges.append((left_neighbor, right_neighbor))
        predicted_keys.append((left_neighbor, right_neighbor))
    coords = {
        left_neighbor: (0.0, -1.0, 0.0),
        u: (0.0, 0.0, 0.0),
        middle: (1.0, 0.0, 0.0),
        v: (2.0, 0.0, 0.0),
        right_neighbor: (3.0, 0.0, 0.0),
        alt_neighbor: (-1.0, 0.0, 0.0),
        9999: (1000.0, 1000.0, 1000.0),
    }
    vertex_count = max(max(vertex for edge in edges for vertex in edge), 9999) + 1
    return FakeMesh(edges=edges, vertex_count=vertex_count, coords=coords), predicted_keys, path


def rank_review_allowed_report(rank, path, *, human_labels=(), duplicate=False, weak=False, selected=False):
    u, middle, v = path
    tier = 3 if weak else 1
    q_floor = 0.3 if weak else 0.8
    q_sum = 0.8 if weak else 1.7
    straightness = 0.3 if weak else 0.9
    return {
        'rank': rank,
        'rank_v2': rank,
        'path_vertex_ids': [u, middle, v],
        'path_edge_keys': [[min(u, middle), max(u, middle)], [min(middle, v), max(middle, v)]],
        'path_edge_indices_blender': [rank * 2, rank * 2 + 1],
        'endpoint_pair_key': [min(u, v), max(u, v)],
        'selected_for_marking': bool(selected),
        'marked': bool(selected),
        'skipped_reason': 'selected' if selected else (
            'duplicate_endpoint_pair_suppressed' if duplicate else 'over_cap_ranked_below_threshold'
        ),
        'duplicate_endpoint_pair_suppressed': bool(duplicate),
        'conflict_reason': None,
        'continuity_tier': tier,
        'q_floor': q_floor,
        'q_sum': q_sum,
        'total_path_length': 0.01 * rank,
        'endpoint_distance': 0.008 * rank,
        'path_straightness': straightness,
        'endpoint_tangent_alignment_u': q_floor,
        'endpoint_tangent_alignment_v': q_floor,
        'min_endpoint_tangent_alignment': q_floor,
        'degree_pattern': (1, 0, 1),
        'component_ids_before': [rank, rank + 100],
        'human_gap_match_labels': list(human_labels),
        'old_validation_target_match_label': None,
    }


def build_rank_review_result(seam_mapping):
    reports = [
        rank_review_allowed_report(rank, (1000 + rank * 10, 1001 + rank * 10, 1002 + rank * 10), selected=True)
        for rank in range(1, 9)
    ]
    reports.extend([
        rank_review_allowed_report(9, (5149, 3003, 3005), human_labels=('8a',), selected=True),
        rank_review_allowed_report(10, (5149, 5103, 3005), human_labels=('8b',), duplicate=True),
        rank_review_allowed_report(11, (3006, 3008, 3039), human_labels=('9',), weak=True),
    ])
    reports.extend(
        rank_review_allowed_report(rank, (2000 + rank * 10, 2001 + rank * 10, 2002 + rank * 10))
        for rank in range(12, 17)
    )
    residual_paths = [
        {
            'label': '8a',
            'path_vertex_ids': [5149, 3003, 3005],
            'candidate_class_phase2e': 'already_marked_but_human_still_sees_gap',
            'recommended_followup': 'review_phase_2b1_rank_9_to_16',
        },
        {
            'label': '8b',
            'path_vertex_ids': [5149, 5103, 3005],
            'candidate_class_phase2e': 'phase_2b1_duplicate_suppressed',
            'recommended_followup': 'review_phase_2b1_rank_9_to_16',
        },
        {
            'label': '9',
            'path_vertex_ids': [3006, 3008, 3039],
            'candidate_class_phase2e': 'phase_2b1_rank_below_cap',
            'recommended_followup': 'review_phase_2b1_rank_9_to_16',
        },
    ]
    return seam_mapping.SeamApplyResult(
        requested=0,
        unique=0,
        applied=0,
        ignored_non_original=0,
        duplicates_skipped=0,
        blender_two_edge_endpoint_bridge_selection_policy='top_k_ranked_continuity_tier_v2',
        blender_two_edge_endpoint_bridge_safety_cap=9,
        blender_two_edge_endpoint_bridge_selected_rank_threshold=9,
        blender_two_edge_endpoint_bridge_raw_allowed_total=16,
        blender_two_edge_endpoint_bridge_deduplicated_allowed_total=15,
        blender_two_edge_endpoint_bridge_paths_marked=9,
        blender_two_edge_endpoint_bridge_allowed_candidate_reports=tuple(reports),
        residual_gap_phase2e_debug={
            'summary': {'residual_paths_total': 3},
            'paths': residual_paths,
            'read_only': True,
        },
    )


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

    def test_apply_seam_keys_uses_editable_gap_fill_not_legacy_repair_stack(self):
        seam_mapping = load_module('uvsp_editable_gap_routing_smoke', ADDON_DIR / 'seam_mapping.py')

        def fail_legacy(*args, **kwargs):
            raise AssertionError('legacy repair function should not be called')

        seam_mapping.apply_two_edge_local_continuity_repair = fail_legacy
        seam_mapping.apply_curved_two_edge_endpoint_bridge_repair = fail_legacy
        seam_mapping.apply_tangent_audit_endpoint_bridge_rescue = fail_legacy

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
        self.assertFalse(result.blender_two_edge_repair_enabled)
        self.assertFalse(result.blender_two_edge_endpoint_bridge_enabled)
        self.assertFalse(result.blender_curved_two_edge_endpoint_bridge_enabled)
        self.assertFalse(result.blender_tangent_audit_endpoint_bridge_enabled)

    def test_debug_sidecar_property_defaults_off_and_is_exposed(self):
        properties_source = read_addon_file('properties.py')
        ui_source = read_addon_file('ui.py')
        operators_source = read_addon_file('operators.py')

        self.assertIn('postprocess_write_debug_sidecars', properties_source)
        self.assertIn("name='Write Legacy Debug Sidecars'", properties_source)
        self.assertIn('default=False', properties_source)
        self.assertIn(
            "description='Debug only: write legacy post-processing diagnostic JSON sidecars.'",
            properties_source,
        )
        self.assertIn("legacy_box.label(text='Legacy / Debug')", ui_source)
        self.assertIn("legacy_box.prop(settings, 'postprocess_write_debug_sidecars')", ui_source)
        self.assertIn(
            'postprocess_write_debug_sidecars=settings.postprocess_write_debug_sidecars',
            operators_source,
        )
        self.assertIn(
            'collect_debug_diagnostics=self._run_settings.postprocess_write_debug_sidecars',
            operators_source,
        )

    def test_editable_gap_hops_property_uses_soft_recommended_max_only(self):
        properties_source = read_addon_file('properties.py')
        start = properties_source.index('postprocess_fill_gap_max_hops')
        end = properties_source.index('postprocess_write_debug_sidecars')
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

    def test_apply_seam_keys_skips_legacy_debug_collectors_by_default(self):
        seam_mapping = load_module('uvsp_debug_collectors_default_off_smoke', ADDON_DIR / 'seam_mapping.py')

        def fail_debug_collector(*args, **kwargs):
            raise AssertionError('legacy debug collector should not be called by default')

        seam_mapping.classify_human_gap_regressions = fail_debug_collector
        seam_mapping.classify_residual_gap_phase2e = fail_debug_collector
        seam_mapping.collect_general_residual_candidates_phase2h = fail_debug_collector
        seam_mapping.simulate_unified_local_continuity_phase2h_r = fail_debug_collector
        seam_mapping.build_phase2h_r3_visual_review = fail_debug_collector
        seam_mapping.simulate_phase2j_r_small_gap_rule = fail_debug_collector
        seam_mapping.simulate_phase2k_r_tangent_audit_rescue = fail_debug_collector

        result = seam_mapping.apply_seam_keys(
            FakeMesh(edges=[(0, 1)], vertex_count=2),
            [(0, 1)],
            clear_existing=True,
        )

        self.assertIsNone(result.human_gap_classification)
        self.assertIsNone(result.residual_gap_phase2e_debug)
        self.assertIsNone(result.general_residual_candidates_phase2h)
        self.assertIsNone(result.unified_local_continuity_simulation_phase2h_r)
        self.assertIsNone(result.phase2h_r3_visual_review)
        self.assertIsNone(result.phase2j_r_small_gap_rule_simulation)
        self.assertIsNone(result.phase2k_r_tangent_audit_rescue)

    def test_apply_seam_keys_debug_collectors_are_reachable_without_legacy_active_repair(self):
        seam_mapping = load_module('uvsp_debug_collectors_enabled_smoke', ADDON_DIR / 'seam_mapping.py')

        def fail_legacy(*args, **kwargs):
            raise AssertionError('debug mode should not call legacy active repair')

        calls = []

        def record(name, payload):
            def _inner(*args, **kwargs):
                calls.append(name)
                return payload
            return _inner

        seam_mapping.apply_two_edge_local_continuity_repair = fail_legacy
        seam_mapping.apply_curved_two_edge_endpoint_bridge_repair = fail_legacy
        seam_mapping.apply_tangent_audit_endpoint_bridge_rescue = fail_legacy
        seam_mapping.classify_human_gap_regressions = record('human', {'name': 'human'})
        seam_mapping.classify_residual_gap_phase2e = record('phase2e', {'name': 'phase2e'})
        seam_mapping.collect_general_residual_candidates_phase2h = record('phase2h', {'name': 'phase2h'})
        seam_mapping.simulate_unified_local_continuity_phase2h_r = record('phase2hr', {'name': 'phase2hr'})
        seam_mapping.build_phase2h_r3_visual_review = record('phase2hr3', {'name': 'phase2hr3'})
        seam_mapping.simulate_phase2j_r_small_gap_rule = record('phase2j', {'name': 'phase2j'})
        seam_mapping.simulate_phase2k_r_tangent_audit_rescue = record('phase2k', {'name': 'phase2k'})

        result = seam_mapping.apply_seam_keys(
            FakeMesh(edges=[(0, 1)], vertex_count=2),
            [(0, 1)],
            clear_existing=True,
            enable_local_repair=True,
            collect_debug_diagnostics=True,
        )

        self.assertEqual(
            calls,
            ['human', 'phase2e', 'phase2h', 'phase2hr', 'phase2hr3', 'phase2j', 'phase2k'],
        )
        self.assertEqual(result.human_gap_classification, {'name': 'human'})
        self.assertEqual(result.phase2k_r_tangent_audit_rescue, {'name': 'phase2k'})
        self.assertFalse(result.blender_two_edge_repair_enabled)
        self.assertFalse(result.blender_two_edge_endpoint_bridge_enabled)
        self.assertFalse(result.blender_curved_two_edge_endpoint_bridge_enabled)
        self.assertFalse(result.blender_tangent_audit_endpoint_bridge_enabled)

    def test_operator_legacy_sidecar_writers_are_guarded_by_debug_setting(self):
        operators_source = read_addon_file('operators.py')
        guard = 'if self._run_settings.postprocess_write_debug_sidecars:'
        guarded_source = operators_source[operators_source.index(guard):]
        writer_names = (
            'write_bridge_apply_debug',
            'write_human_gap_classification',
            'write_residual_gap_phase2e_debug',
            'write_endpoint_bridge_ranking_debug',
            'write_rank_9_to_16_review',
            'write_general_residual_candidates_phase2h',
            'write_unified_local_continuity_simulation_phase2h_r',
            'write_phase2h_r3_visual_review',
            'write_phase2j_r_small_gap_rule_simulation',
            'write_phase2k_r_tangent_audit_rescue',
        )

        for writer_name in writer_names:
            self.assertIn(f'seam_mapping.{writer_name}', guarded_source)
            self.assertNotIn(f'seam_mapping.{writer_name}', operators_source[:operators_source.index(guard)])

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

    def test_active_repair_signatures_do_not_accept_hardcoded_paths(self):
        seam_mapping = load_module('uvsp_seam_mapping_repair_signature_smoke', ADDON_DIR / 'seam_mapping.py')

        self.assertNotIn(
            'target_paths',
            inspect.signature(seam_mapping.apply_two_edge_local_continuity_repair).parameters,
        )

    def test_production_hardcoded_path_exception_telemetry_is_disabled(self):
        seam_mapping = load_module('uvsp_seam_mapping_hardcode_telemetry_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1)], vertex_count=2)

        result = seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1)],
            clear_existing=True,
            enable_local_repair=True,
        )

        self.assertFalse(result.production_hardcoded_path_exceptions_enabled)
        self.assertTrue(result.production_hardcoded_path_exceptions_removed)
        self.assertTrue(result.diagnostic_path_labels_read_only)
        self.assertFalse(result.human_case_over_cap_exception_used)
        self.assertFalse(result.target_path_2045_2541_4884_accepted_by_target_over_cap_exception)
        self.assertFalse(result.target_path_2540_2541_2544_accepted_by_target_over_cap_exception)

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

    def test_endpoint_bridge_ranking_debug_sidecar_is_read_only_and_has_no_probabilities(self):
        seam_mapping = load_module('uvsp_endpoint_bridge_ranking_sidecar_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_endpoint_bridge_mesh()
        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )
        flags_before = [edge.use_seam for edge in mesh.edges]
        debug = seam_mapping.build_endpoint_bridge_ranking_debug(result)
        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path = str(Path(temp_dir) / 'prediction.json')
            Path(prediction_path).write_text('{}', encoding='utf-8')
            sidecar = seam_mapping.write_endpoint_bridge_ranking_debug(prediction_path, result)
            payload = json.loads(Path(sidecar).read_text(encoding='utf-8'))

        self.assertTrue(sidecar.endswith('_endpoint_bridge_ranking_debug.json'))
        self.assertTrue(debug['read_only'])
        self.assertEqual([edge.use_seam for edge in mesh.edges], flags_before)
        self.assertNotIn('probability', json.dumps(payload).lower())

    def test_human_gap_classifier_reports_editable_and_missing_edges(self):
        seam_mapping = load_module('uvsp_gap_classifier_edges_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2)], vertex_count=4)

        classification = seam_mapping.classify_human_gap_regressions(
            mesh,
            paths=(
                ('editable', 'editable', 'main', (0, 1, 2)),
                ('missing', 'missing', 'main', (0, 3)),
            ),
        )

        editable, missing = classification['paths']
        self.assertTrue(editable['all_edges_exist_in_blender'])
        self.assertFalse(missing['all_edges_exist_in_blender'])
        self.assertEqual(missing['candidate_class'], 'non_original_or_missing_blender_edge')
        self.assertEqual(missing['rejection_reason'], 'edge_not_found')

    def test_human_gap_classifier_reports_already_marked_path(self):
        seam_mapping = load_module('uvsp_gap_classifier_marked_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2)], vertex_count=3)
        seam_mapping.apply_seam_keys(mesh, [(0, 1), (1, 2)], clear_existing=True)
        flags_before = [edge.use_seam for edge in mesh.edges]

        classification = seam_mapping.classify_human_gap_regressions(
            mesh,
            paths=(('marked', 'marked', 'main', (0, 1, 2)),),
            predicted_keys={(0, 1), (1, 2)},
        )

        report = classification['paths'][0]
        self.assertEqual(report['candidate_class'], 'already_marked')
        self.assertTrue(report['already_all_marked'])
        self.assertTrue(report['marked_by_prediction_if_traceable'])
        self.assertEqual([edge.use_seam for edge in mesh.edges], flags_before)

    def test_human_gap_classifier_classifies_one_edge_missing_continuity(self):
        seam_mapping = load_module('uvsp_gap_classifier_one_edge_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_degree_pattern_mesh(1, 2, key=(100, 200))
        seam_mapping.apply_seam_keys(mesh, predicted_keys, clear_existing=True, enable_local_repair=False)

        classification = seam_mapping.classify_human_gap_regressions(
            mesh,
            paths=(('one', 'one', 'main', (100, 200)),),
        )

        report = classification['paths'][0]
        self.assertEqual(report['candidate_class'], 'phase_2a1_one_edge_missing_continuity')
        self.assertTrue(report['would_be_allowed_by_phase_2a1'])

    def test_human_gap_classifier_classifies_two_edge_same_component(self):
        seam_mapping = load_module('uvsp_gap_classifier_two_same_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_two_edge_repair_mesh((2, 2))
        seam_mapping.apply_seam_keys(mesh, predicted_keys, clear_existing=True, enable_local_repair=False)

        classification = seam_mapping.classify_human_gap_regressions(
            mesh,
            paths=(('same', 'same', 'main', (100, 101, 102)),),
        )

        report = classification['paths'][0]
        self.assertEqual(report['candidate_class'], 'phase_2b_same_component_two_edge')
        self.assertTrue(report['would_be_allowed_by_phase_2b_same_component'])

    def test_human_gap_classifier_classifies_two_edge_endpoint_bridge(self):
        seam_mapping = load_module('uvsp_gap_classifier_endpoint_bridge_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_endpoint_bridge_mesh()
        seam_mapping.apply_seam_keys(mesh, predicted_keys, clear_existing=True, enable_local_repair=False)

        classification = seam_mapping.classify_human_gap_regressions(
            mesh,
            paths=(('bridge', 'bridge', 'main', (100, 101, 102)),),
        )

        report = classification['paths'][0]
        self.assertEqual(report['candidate_class'], 'phase_2b1_inter_component_two_edge_endpoint_bridge')
        self.assertTrue(report['would_be_allowed_by_phase_2b1_endpoint_bridge'])
        self.assertTrue(all(report['tangent_available_flags']))

    def test_human_gap_classifier_classifies_three_edge_and_endpoint_to_skeleton(self):
        seam_mapping = load_module('uvsp_gap_classifier_unsupported_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (2, 3), (10, 11), (11, 12), (10, 20), (11, 21)])
        seam_mapping.apply_seam_keys(mesh, [(10, 20), (11, 21)], clear_existing=True, enable_local_repair=False)

        classification = seam_mapping.classify_human_gap_regressions(
            mesh,
            paths=(
                ('three', 'three', 'main', (0, 1, 2, 3)),
                ('junction', 'junction', 'main', (10, 11, 12)),
            ),
        )

        self.assertEqual(classification['paths'][0]['candidate_class'], 'three_edge_local_bridge')
        self.assertEqual(classification['paths'][0]['rejection_reason'], 'path_length_not_supported')
        self.assertEqual(
            classification['paths'][1]['candidate_class'],
            'endpoint_to_skeleton_or_near_junction',
        )

    def test_human_gap_classifier_reports_over_cap_skip_from_existing_reports(self):
        seam_mapping = load_module('uvsp_gap_classifier_over_cap_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_many_two_edge_repair_candidates(17)
        seam_mapping.apply_seam_keys(mesh, predicted_keys, clear_existing=True, enable_local_repair=False)
        repair = seam_mapping.apply_two_edge_local_continuity_repair(mesh, enabled=True)

        classification = seam_mapping.classify_human_gap_regressions(
            mesh,
            paths=(('cap', 'cap', 'main', (6000, 6001, 6002)),),
            two_edge_reports=repair['candidate_reports'],
        )

        report = classification['paths'][0]
        self.assertTrue(report['skipped_only_due_to_over_cap'])
        self.assertEqual(report['phase_2b_rejection_reason'], 'repair_over_cap')
        self.assertEqual(report['candidate_class'], 'phase_2b_same_component_two_edge')

    def test_human_gap_classifier_summary_and_recommendation(self):
        seam_mapping = load_module('uvsp_gap_classifier_summary_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_many_two_edge_repair_candidates(17)
        seam_mapping.apply_seam_keys(mesh, predicted_keys, clear_existing=True, enable_local_repair=False)
        repair = seam_mapping.apply_two_edge_local_continuity_repair(mesh, enabled=True)
        paths = tuple(
            (str(index), 'same', str(index), (6000 + index * 20, 6001 + index * 20, 6002 + index * 20))
            for index in range(10)
        )

        classification = seam_mapping.classify_human_gap_regressions(
            mesh,
            paths=paths,
            two_edge_reports=repair['candidate_reports'],
        )

        summary = classification['summary']
        self.assertEqual(summary['total_paths_classified'], 10)
        self.assertEqual(summary['count_skipped_only_due_to_over_cap'], 10)
        self.assertEqual(
            summary['recommended_next_action'],
            'improve_phase_2b_same_component_ranking',
        )

    def test_human_gap_classifier_writes_sidecar_and_operator_flow_references_it(self):
        seam_mapping = load_module('uvsp_gap_classifier_sidecar_smoke', ADDON_DIR / 'seam_mapping.py')
        result = seam_mapping.SeamApplyResult(
            requested=0,
            unique=0,
            applied=0,
            ignored_non_original=0,
            duplicates_skipped=0,
            human_gap_classification={
                'summary': {
                    'total_paths_classified': 1,
                    'recommended_next_action': 'no_dominant_class',
                },
                'paths': [],
                'read_only': True,
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path = str(Path(temp_dir) / 'prediction.json')
            Path(prediction_path).write_text('{}', encoding='utf-8')
            sidecar = seam_mapping.write_human_gap_classification(prediction_path, result)
            payload = json.loads(Path(sidecar).read_text(encoding='utf-8'))

        self.assertTrue(sidecar.endswith('_human_gap_classification.json'))
        self.assertTrue(payload['read_only'])
        self.assertIn('write_human_gap_classification', read_addon_file('operators.py'))

    def test_phase2e_residual_sidecar_and_already_marked_classification(self):
        seam_mapping = load_module('uvsp_phase2e_sidecar_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(234, 319), (319, 318), (318, 214)], vertex_count=400)
        result = seam_mapping.apply_seam_keys(
            mesh,
            [(234, 319), (319, 318), (318, 214)],
            clear_existing=True,
            enable_local_repair=True,
            collect_debug_diagnostics=True,
        )
        flags_before = [edge.use_seam for edge in mesh.edges]
        report = next(item for item in result.residual_gap_phase2e_debug['paths'] if item['label'] == '2')

        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path = str(Path(temp_dir) / 'prediction.json')
            Path(prediction_path).write_text('{}', encoding='utf-8')
            sidecar = seam_mapping.write_residual_gap_phase2e_debug(prediction_path, result)
            payload = json.loads(Path(sidecar).read_text(encoding='utf-8'))

        self.assertTrue(sidecar.endswith('_residual_gap_phase2e_debug.json'))
        self.assertTrue(payload['read_only'])
        self.assertEqual(report['candidate_class_phase2e'], 'already_marked_but_human_still_sees_gap')
        self.assertTrue(report['is_visual_or_apply_verification_issue'])
        self.assertEqual([edge.use_seam for edge in mesh.edges], flags_before)
        self.assertIn('write_residual_gap_phase2e_debug', read_addon_file('operators.py'))

    def test_phase2e_residual_classifies_new_repair_classes_and_missing_edge(self):
        seam_mapping = load_module('uvsp_phase2e_new_classes_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (234, 319), (319, 318), (318, 214),
                (2391, 2000), (2391, 1723),
                (1734, 1700), (1700, 1722), (1734, 1800), (1722, 1801),
                (1734, 1723), (1723, 1722),
            ],
            vertex_count=2500,
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(2391, 2000), (1734, 1700), (1700, 1722), (1734, 1800), (1722, 1801)],
            clear_existing=True,
            enable_local_repair=False,
        )

        classification = seam_mapping.classify_residual_gap_phase2e(mesh)
        by_label = {report['label']: report for report in classification['paths']}

        self.assertEqual(by_label['2']['candidate_class_phase2e'], 'three_edge_local_bridge')
        self.assertEqual(
            by_label['15a']['candidate_class_phase2e'],
            'endpoint_to_skeleton_or_near_junction',
        )
        self.assertTrue(by_label['15a']['one_endpoint_is_non_seam'])
        self.assertEqual(by_label['15a']['why_phase_2a1_does_not_apply'], 'endpoint_not_seam_vertex')
        self.assertEqual(
            by_label['15b']['candidate_class_phase2e'],
            'same_component_two_edge_local_bridge',
        )
        self.assertTrue(by_label['15b']['same_component_status'])
        self.assertEqual(by_label['15b']['why_phase_2b1_rejects_it'], 'endpoint_not_degree_1')
        self.assertEqual(
            by_label['14a']['candidate_class_phase2e'],
            'non_original_or_missing_blender_edge',
        )

    def test_phase2f_rank_review_sidecar_and_rank_window(self):
        seam_mapping = load_module('uvsp_phase2f_sidecar_smoke', ADDON_DIR / 'seam_mapping.py')
        result = build_rank_review_result(seam_mapping)
        payload = seam_mapping.build_rank_9_to_16_review(result)

        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path = str(Path(temp_dir) / 'prediction.json')
            Path(prediction_path).write_text('{}', encoding='utf-8')
            sidecar = seam_mapping.write_rank_9_to_16_review(prediction_path, result)
            sidecar_payload = json.loads(Path(sidecar).read_text(encoding='utf-8'))

        self.assertTrue(sidecar.endswith('_rank_9_to_16_review.json'))
        self.assertTrue(payload['read_only'])
        self.assertTrue(sidecar_payload['read_only'])
        self.assertEqual(
            [report['rank_v2'] for report in payload['rank_9_to_16_candidates']],
            list(range(9, 17)),
        )
        self.assertIn('write_rank_9_to_16_review', read_addon_file('operators.py'))

    def test_phase2f_hypothetical_cap_summaries_and_duplicate_exclusion(self):
        seam_mapping = load_module('uvsp_phase2f_caps_smoke', ADDON_DIR / 'seam_mapping.py')
        payload = seam_mapping.build_rank_9_to_16_review(build_rank_review_result(seam_mapping))
        summaries = {
            item['hypothetical_cap']: item
            for item in payload['hypothetical_cap_summaries']
        }

        self.assertEqual(set(summaries), {8, 9, 10, 12, 16})
        self.assertEqual(summaries[9]['risk_summary'], 'current')
        self.assertEqual(summaries[9]['additional_candidates_selected'], 0)
        self.assertEqual(summaries[8]['additional_candidates_selected'], 0)
        self.assertEqual(summaries[10]['additional_candidates_selected'], 1)
        self.assertEqual(summaries[10]['additional_duplicate_candidates_selected'], 0)
        self.assertNotIn([5149, 5103, 3005], summaries[10]['path_vertex_ids_added'])

    def test_phase2f_review_classifies_rank9_duplicate_weak_and_nonhuman(self):
        seam_mapping = load_module('uvsp_phase2f_classes_smoke', ADDON_DIR / 'seam_mapping.py')
        payload = seam_mapping.build_rank_9_to_16_review(build_rank_review_result(seam_mapping))
        by_rank = {
            report['rank_v2']: report
            for report in payload['rank_9_to_16_candidates']
        }

        self.assertEqual(by_rank[9]['candidate_review_class'], 'strong_human_rank_below_cap')
        self.assertTrue(by_rank[9]['would_be_selected_if_cap_9'])
        self.assertTrue(by_rank[9]['selected_for_marking'])
        self.assertEqual(by_rank[9]['visual_review_priority'], 'high')
        self.assertEqual(by_rank[10]['candidate_review_class'], 'duplicate_alternative')
        self.assertFalse(by_rank[10]['would_be_selected_if_cap_10'])
        self.assertEqual(by_rank[11]['candidate_review_class'], 'weak_geometry_rank_below_cap')
        self.assertEqual(by_rank[12]['candidate_review_class'], 'non_human_rank_below_cap')

    def test_phase2f_special_5149_report_and_recommendation(self):
        seam_mapping = load_module('uvsp_phase2f_special_smoke', ADDON_DIR / 'seam_mapping.py')
        payload = seam_mapping.build_rank_9_to_16_review(build_rank_review_result(seam_mapping))
        special = payload['special_reports']['path_5149_3003_3005']
        summary = payload['summary']

        self.assertTrue(special['found_in_review'])
        self.assertEqual(special['rank_v2'], 9)
        self.assertEqual(special['continuity_tier'], 1)
        self.assertEqual(special['rank_delta_from_cap'], 0)
        self.assertFalse(special['is_highest_ranked_unselected_human_candidate'])
        self.assertTrue(special['would_be_selected_if_cap_9'])
        self.assertFalse(special['duplicate_endpoint_pair_suppressed'])
        self.assertEqual(special['visual_review_priority'], 'high')
        self.assertEqual(summary['recommended_next_action'], 'do_not_increase_cap_due_to_weak_candidates')
        self.assertEqual(summary['human_matched_review_candidates'], 3)
        self.assertEqual(summary['duplicate_suppressed_review_candidates'], 1)
        self.assertEqual(summary['weak_geometry_review_candidates'], 1)

    def test_phase2f_review_is_read_only_for_mesh_flags(self):
        seam_mapping = load_module('uvsp_phase2f_readonly_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_endpoint_bridge_mesh()
        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )
        flags_before = [edge.use_seam for edge in mesh.edges]

        seam_mapping.build_rank_9_to_16_review(result)

        self.assertEqual([edge.use_seam for edge in mesh.edges], flags_before)

    def test_phase2h_sidecar_is_emitted_and_read_only(self):
        seam_mapping = load_module('uvsp_phase2h_sidecar_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2)], vertex_count=3)
        result = seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1)],
            clear_existing=True,
            enable_local_repair=False,
        )
        flags_before = [edge.use_seam for edge in mesh.edges]

        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path = str(Path(temp_dir) / 'prediction.json')
            Path(prediction_path).write_text('{}', encoding='utf-8')
            sidecar = seam_mapping.write_general_residual_candidates_phase2h(prediction_path, result)
            payload = json.loads(Path(sidecar).read_text(encoding='utf-8'))

        self.assertTrue(sidecar.endswith('_general_residual_candidates_phase2h.json'))
        self.assertTrue(payload['read_only'])
        self.assertEqual([edge.use_seam for edge in mesh.edges], flags_before)
        self.assertIn('write_general_residual_candidates_phase2h', read_addon_file('operators.py'))

    def test_phase2h_collects_length1_length2_length3_and_missing_residuals(self):
        seam_mapping = load_module('uvsp_phase2h_classes_smoke', ADDON_DIR / 'seam_mapping.py')
        edges = [
            (0, 1), (2, 3), (1, 2),
            (10, 11), (13, 14), (11, 12), (12, 13),
            (20, 21), (21, 22), (20, 23), (22, 24), (20, 25), (25, 22),
            (30, 31), (34, 35), (31, 32), (32, 33), (33, 34),
            (40, 41), (41, 42),
        ]
        coords = {
            10: (-0.01, 0.0, 0.0), 11: (0.0, 0.0, 0.0), 12: (0.01, 0.0, 0.0),
            13: (0.02, 0.0, 0.0), 14: (0.03, 0.0, 0.0),
            20: (0.0, 1.0, 0.0), 21: (0.01, 1.01, 0.0), 22: (0.02, 1.0, 0.0),
            23: (-0.01, 1.0, 0.0), 24: (0.03, 1.0, 0.0), 25: (0.01, 1.0, 0.0),
            30: (0.0, 2.0, 0.0), 31: (0.01, 2.0, 0.0), 32: (0.02, 2.0, 0.0),
            33: (0.03, 2.0, 0.0), 34: (0.04, 2.0, 0.0), 35: (0.05, 2.0, 0.0),
            9999: (1.0, 1.0, 1.0),
        }
        mesh = FakeMesh(edges=edges, vertex_count=10000, coords=coords)
        seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1), (2, 3), (10, 11), (13, 14), (20, 23), (22, 24), (30, 31), (34, 35), (40, 41)],
            clear_existing=True,
            enable_local_repair=False,
        )
        residual_payload = {
            'paths': [
                {'label': 'one', 'path_vertex_ids': [1, 2], 'candidate_class_phase2e': 'one'},
                {'label': 'bridge', 'path_vertex_ids': [11, 12, 13], 'candidate_class_phase2e': 'bridge'},
                {'label': 'duplicate', 'path_vertex_ids': [20, 21, 22], 'candidate_class_phase2e': 'duplicate'},
                {'label': 'three', 'path_vertex_ids': [31, 32, 33, 34], 'candidate_class_phase2e': 'three'},
                {'label': 'missing', 'path_vertex_ids': [70, 71], 'candidate_class_phase2e': 'missing', 'all_edges_exist_in_blender': False},
            ],
            'read_only': True,
        }
        endpoint_reports = (
            {'path_vertex_ids': [20, 21, 22], 'duplicate_endpoint_pair_suppressed': True, 'rejection_reason': None, 'rank_v2': 7},
        )

        payload = seam_mapping.collect_general_residual_candidates_phase2h(
            mesh,
            predicted_keys={(0, 1), (2, 3), (10, 11), (13, 14), (20, 23), (22, 24), (30, 31), (34, 35), (40, 41)},
            endpoint_bridge_reports=endpoint_reports,
            residual_payload=residual_payload,
        )
        by_path = {
            tuple(report['path_vertex_ids']): report
            for report in payload['candidates']
        }

        self.assertEqual(by_path[(1, 2)]['candidate_class_phase2h'], 'one_edge_missing_continuity')
        self.assertEqual(
            by_path[(11, 12, 13)]['candidate_class_phase2h'],
            'two_edge_inter_component_endpoint_bridge',
        )
        self.assertEqual(by_path[(20, 21, 22)]['candidate_class_phase2h'], 'two_edge_duplicate_alternative')
        self.assertEqual(by_path[(31, 32, 33, 34)]['candidate_class_phase2h'], 'three_edge_local_bridge')
        self.assertEqual(by_path[(70, 71)]['candidate_class_phase2h'], 'non_original_or_missing_blender_edge')
        self.assertGreaterEqual(payload['summary']['candidates_by_path_length'][1], 1)
        self.assertGreaterEqual(payload['summary']['candidates_by_path_length'][2], 1)
        self.assertGreaterEqual(payload['summary']['candidates_by_path_length'][3], 1)

    def test_phase2h_classifies_same_component_endpoint_to_skeleton_and_tangent_failed(self):
        seam_mapping = load_module('uvsp_phase2h_topology_smoke', ADDON_DIR / 'seam_mapping.py')
        coords = {
            100: (0.0, 0.0, 0.0), 101: (0.01, 0.0, 0.0), 102: (0.02, 0.0, 0.0),
            103: (0.03, 0.0, 0.0), 104: (0.04, 0.0, 0.0), 105: (0.05, 0.0, 0.0),
            200: (0.0, 1.0, 0.0), 201: (0.01, 1.0, 0.0), 202: (0.02, 1.0, 0.0),
            210: (0.0, 2.0, 0.0), 211: (0.01, 2.0, 0.0), 212: (0.02, 2.0, 0.0),
            213: (0.03, 2.0, 0.0), 9999: (1.0, 1.0, 1.0),
        }
        mesh = FakeMesh(
            edges=[
                (100, 101), (104, 105), (100, 105), (101, 102), (102, 104),
                (200, 201), (201, 202),
                (210, 211), (212, 213), (211, 202), (202, 212),
            ],
            vertex_count=10000,
            coords=coords,
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(100, 101), (104, 105), (100, 105), (200, 201), (210, 211), (212, 213)],
            clear_existing=True,
            enable_local_repair=False,
        )
        residual_payload = {
            'paths': [
                {'label': 'same', 'path_vertex_ids': [101, 102, 104], 'candidate_class_phase2e': 'same'},
                {'label': 'junction', 'path_vertex_ids': [200, 201, 202], 'candidate_class_phase2e': 'junction'},
                {'label': 'tangent', 'path_vertex_ids': [211, 202, 212], 'candidate_class_phase2e': 'tangent'},
            ],
            'read_only': True,
        }
        endpoint_reports = (
            {'path_vertex_ids': [211, 202, 212], 'rejection_reason': 'tangent_alignment_failed', 'rank_v2': None},
        )

        payload = seam_mapping.collect_general_residual_candidates_phase2h(
            mesh,
            endpoint_bridge_reports=endpoint_reports,
            residual_payload=residual_payload,
        )
        by_path = {tuple(report['path_vertex_ids']): report for report in payload['candidates']}

        self.assertEqual(
            by_path[(101, 102, 104)]['candidate_class_phase2h'],
            'two_edge_same_component_local_closure',
        )
        self.assertEqual(
            by_path[(200, 201, 202)]['candidate_class_phase2h'],
            'two_edge_endpoint_to_skeleton_or_near_junction',
        )
        self.assertEqual(
            by_path[(211, 202, 212)]['candidate_class_phase2h'],
            'two_edge_tangent_failed_endpoint_bridge',
        )

    def test_phase2h_residual_mapping_recommendations_and_truncation(self):
        seam_mapping = load_module('uvsp_phase2h_summary_smoke', ADDON_DIR / 'seam_mapping.py')
        edges = []
        residual_paths = []
        for index in range(12):
            base = index * 10
            edges.extend([(base, base + 1), (base + 1, base + 2)])
        residual_paths.append({
            'label': 'truncated_residual',
            'path_vertex_ids': [110, 111, 112],
            'candidate_class_phase2e': 'unknown',
        })
        mesh = FakeMesh(edges=edges, vertex_count=200)

        payload = seam_mapping.collect_general_residual_candidates_phase2h(
            mesh,
            residual_payload={'paths': residual_paths, 'read_only': True},
        )

        self.assertGreater(payload['summary']['per_class_truncation_counts']['unsupported_or_unknown'], 0)
        mapping = payload['human_residual_mapping'][0]
        self.assertEqual(mapping['residual_label'], 'truncated_residual')
        self.assertTrue(mapping['matched_candidate_ids'])
        self.assertFalse(mapping['candidate_generation_cap_truncated'])
        self.assertIn(mapping['best_matching_candidate_id'], {
            report['candidate_id'] for report in payload['candidates']
        })

    def test_phase2h_recommendation_heuristics_for_mixed_and_dominant_classes(self):
        seam_mapping = load_module('uvsp_phase2h_recommendation_smoke', ADDON_DIR / 'seam_mapping.py')
        mixed_summary = {
            'residual_paths_total': 3,
            'residual_coverage_by_class': {
                'three_edge_local_bridge': 1,
                'one_edge_endpoint_to_skeleton': 1,
                'non_original_or_missing_blender_edge': 1,
            },
        }
        three_summary = {
            'residual_paths_total': 5,
            'residual_coverage_by_class': {'three_edge_local_bridge': 3},
        }
        endpoint_summary = {
            'residual_paths_total': 5,
            'residual_coverage_by_class': {'one_edge_endpoint_to_skeleton': 3},
        }

        self.assertEqual(
            seam_mapping._phase2h_recommendation(mixed_summary),
            'no_single_dominant_next_action',
        )
        self.assertEqual(
            seam_mapping._phase2h_recommendation(three_summary),
            'consider_three_edge_classifier',
        )
        self.assertEqual(
            seam_mapping._phase2h_recommendation(endpoint_summary),
            'consider_endpoint_to_skeleton_classifier',
        )

    def test_phase2h_collection_is_deterministic(self):
        seam_mapping = load_module('uvsp_phase2h_deterministic_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(edges=[(0, 1), (1, 2), (3, 4), (2, 3)], vertex_count=5)
        seam_mapping.apply_seam_keys(mesh, [(0, 1), (3, 4)], clear_existing=True, enable_local_repair=False)

        first = seam_mapping.collect_general_residual_candidates_phase2h(mesh)
        second = seam_mapping.collect_general_residual_candidates_phase2h(mesh)

        self.assertEqual(first['summary'], second['summary'])
        self.assertEqual(first['candidates'], second['candidates'])

    def test_unified_local_continuity_simulation_normalizes_open_residual_labels(self):
        seam_mapping = load_module('uvsp_phase2hr_normalized_labels_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(5561, 5562), (5553, 5554), (5562, 5464), (5464, 5553)],
            vertex_count=6000,
            coords={
                5561: (-0.01, 0.0, 0.0),
                5562: (0.0, 0.0, 0.0),
                5464: (0.01, 0.0, 0.0),
                5553: (0.02, 0.0, 0.0),
                5554: (0.03, 0.0, 0.0),
                9999: (1.0, 1.0, 1.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(5561, 5562), (5553, 5554)],
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_unified_local_continuity_phase2h_r(
            mesh,
            residual_payload={
                'paths': [
                    {
                        'label': '6a',
                        'path_vertex_ids': [5562, 5464, 5553],
                        'candidate_class_phase2e': 'phase_2b1_rank_below_cap',
                    },
                ],
                'read_only': True,
            },
        )
        normalized = payload['normalized_residual_coverage']
        alias_by_label = {
            item['canonical_label']: item
            for item in normalized['residual_alias_groups']
        }
        canonical_labels = {
            item['canonical_label']
            for item in normalized['canonical_open_residual_paths']
        }

        self.assertEqual(normalized['canonical_open_residual_paths_total'], 14)
        self.assertIn('6a', alias_by_label)
        self.assertIn('6a', alias_by_label['6a']['aliases'])
        self.assertIn('residual_6a', alias_by_label['6a']['aliases'])
        self.assertGreaterEqual(normalized['duplicate_label_count'], 14)
        self.assertNotIn('8a', canonical_labels)
        self.assertFalse(any(label.startswith('solved_') for label in canonical_labels))
        self.assertGreaterEqual(normalized['solved_label_count'], 4)
        self.assertNotIn([5149, 3003, 3005], [
            item['path_vertex_ids'] for item in normalized['canonical_open_residual_paths']
        ])
        self.assertNotIn([2557, 2558], [
            item['path_vertex_ids'] for item in normalized['canonical_open_residual_paths']
        ])
        self.assertNotIn([2045, 2541, 4884], [
            item['path_vertex_ids'] for item in normalized['canonical_open_residual_paths']
        ])
        self.assertNotIn([2540, 2541, 2544], [
            item['path_vertex_ids'] for item in normalized['canonical_open_residual_paths']
        ])

    def test_unified_local_continuity_simulation_low_straightness_blocks_readiness(self):
        seam_mapping = load_module('uvsp_phase2hr_low_straightness_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(99, 100), (102, 103), (100, 101), (101, 102)],
            vertex_count=10000,
            coords={
                99: (-1.0, 0.0, 0.0),
                100: (0.0, 0.0, 0.0),
                101: (1.0, 0.0, 0.0),
                102: (1.2, 0.98, 0.0),
                103: (1.4, 1.96, 0.0),
                9999: (10.0, 10.0, 10.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)
        policies = {
            item['policy_name']: item
            for item in payload['policies']
        }
        low_selected = [
            candidate
            for policy in payload['policies']
            for candidate in policy['selected_candidates']
            if candidate['selected_low_straightness_two_edge_bridge']
        ]

        self.assertTrue(low_selected)
        self.assertTrue(low_selected[0]['low_straightness_warning'])
        self.assertEqual(
            low_selected[0]['visual_review_only_reason'],
            'selected endpoint bridge has low path straightness',
        )
        self.assertGreater(
            policies['conservative_length2_only']['selected_low_straightness_two_edge_bridge_total'],
            0,
        )
        self.assertIn(
            'low_straightness_endpoint_bridge_selected',
            policies['conservative_length2_only']['blocking_reasons'],
        )
        self.assertNotEqual(
            policies['conservative_length2_only']['production_readiness'],
            'candidate_for_future_active_design',
        )

    def test_unified_local_continuity_simulation_reports_straight_but_tangent_weak(self):
        seam_mapping = load_module('uvsp_phase2hr_tangent_weak_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(99, 100), (102, 103), (100, 101), (101, 102)],
            vertex_count=10000,
            coords={
                99: (0.0, -1.0, 0.0),
                100: (0.0, 0.0, 0.0),
                101: (1.0, 0.0, 0.0),
                102: (2.0, 0.0, 0.0),
                103: (3.0, 0.0, 0.0),
                9999: (10.0, 10.0, 10.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)
        candidates = payload['straight_but_tangent_weak_candidates']

        self.assertGreaterEqual(payload['straight_but_tangent_weak_candidate_count'], 1)
        self.assertTrue(candidates[0]['straight_but_tangent_weak'])
        self.assertTrue(candidates[0]['possible_tangent_model_false_negative'])
        self.assertTrue(candidates[0]['do_not_select_automatically'])
        self.assertFalse(any(
            candidate['selected_by_simulation']
            for policy in payload['policies']
            for candidate in policy['selected_candidates']
            if candidate['path_vertex_ids'] == candidates[0]['path_vertex_ids']
        ))

    def test_unified_local_continuity_simulation_high_risk_blocks_readiness(self):
        seam_mapping = load_module('uvsp_phase2hr_high_risk_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (3, 4), (1, 2), (2, 3)],
            vertex_count=5,
            coords={
                0: (0.0, 0.0, 0.0),
                1: (0.01, 0.0, 0.0),
                2: (0.02, 0.0, 0.0),
                3: (0.03, 0.0, 0.0),
                4: (0.04, 0.0, 0.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1), (3, 4)],
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)
        probe = next(
            item for item in payload['policies']
            if item['policy_name'] == 'class_balanced_probe'
        )

        self.assertGreater(probe['selected_high_risk_class_total'], 0)
        self.assertIn('high_risk_class_selected', probe['blocking_reasons'])
        self.assertEqual(probe['production_readiness'], 'diagnostic_only')

    def test_unified_local_continuity_simulation_labels_do_not_change_selection(self):
        seam_mapping = load_module('uvsp_phase2hr_label_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_endpoint_bridge_mesh()
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        no_label = seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)
        with_label = seam_mapping.simulate_unified_local_continuity_phase2h_r(
            mesh,
            residual_payload={
                'paths': [
                    {
                        'label': 'manual_label',
                        'path_vertex_ids': [100, 101, 102],
                        'candidate_class_phase2e': 'manual',
                    },
                ],
                'read_only': True,
            },
        )

        def selected_paths(payload):
            return {
                policy['policy_name']: [
                    item['path_vertex_ids'] for item in policy['selected_candidates']
                ]
                for policy in payload['policies']
            }

        self.assertEqual(selected_paths(no_label), selected_paths(with_label))
        labeled = [
            item
            for policy in with_label['policies']
            for item in policy['residual_matched_candidates']
            if 'manual_label' in item['residual_match_labels']
        ]
        self.assertTrue(labeled)
        self.assertNotIn('manual_label', json.dumps(no_label))

    def test_unified_local_continuity_simulation_duplicate_suppression_and_determinism(self):
        seam_mapping = load_module('uvsp_phase2hr_duplicate_smoke', ADDON_DIR / 'seam_mapping.py')
        coords = {
            99: (-0.01, 0.0, 0.0),
            100: (0.0, 0.0, 0.0),
            101: (0.01, 0.0, 0.0),
            102: (0.02, 0.0, 0.0),
            103: (0.03, 0.0, 0.0),
            104: (0.01, 0.01, 0.0),
            9999: (1.0, 1.0, 1.0),
        }
        mesh = FakeMesh(
            edges=[(99, 100), (102, 103), (100, 101), (101, 102), (100, 104), (104, 102)],
            vertex_count=10000,
            coords=coords,
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )

        first = seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)
        second = seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)

        self.assertEqual(first['summary'], second['summary'])
        self.assertEqual(
            [
                [item['candidate_id'] for item in policy['selected_candidates']]
                for policy in first['policies']
            ],
            [
                [item['candidate_id'] for item in policy['selected_candidates']]
                for policy in second['policies']
            ],
        )
        self.assertGreaterEqual(
            sum(policy['duplicate_candidates_suppressed'] for policy in first['policies']),
            1,
        )

    def test_unified_local_continuity_simulation_compact_and_probability_free(self):
        seam_mapping = load_module('uvsp_phase2hr_compact_smoke', ADDON_DIR / 'seam_mapping.py')
        edges = []
        predicted_keys = []
        coords = {9999: (1.0, 1.0, 1.0)}
        for index in range(14):
            append_endpoint_bridge_candidate(edges, predicted_keys, coords, 700 + index * 10, y=float(index))
        mesh = FakeMesh(edges=edges, vertex_count=10000, coords=coords)
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)
        text = json.dumps(payload).lower()

        self.assertTrue(payload['compact_sidecar'])
        self.assertGreater(payload['total_candidates_considered'], 0)
        self.assertGreater(payload['total_candidates_reported'], 0)
        self.assertIn('selected_candidates', payload['policies'][0])
        self.assertIn('residual_matched_candidates', payload['policies'][0])
        self.assertIn('top_rejected_candidates_by_class', payload['policies'][0])
        self.assertIn('per_class_reported_counts', payload)
        self.assertIn('per_class_truncation_counts', payload)
        self.assertNotIn('model_probability', text)
        self.assertNotIn('probability_score', text)
        self.assertFalse(payload['probabilities_used'])

    def test_phase2hr3_visual_review_sidecar_is_emitted_and_read_only(self):
        seam_mapping = load_module('uvsp_phase2hr3_sidecar_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_endpoint_bridge_mesh()
        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )
        flags_before = [edge.use_seam for edge in mesh.edges]

        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path = str(Path(temp_dir) / 'prediction.json')
            Path(prediction_path).write_text('{}', encoding='utf-8')
            sidecar = seam_mapping.write_phase2h_r3_visual_review(prediction_path, result)
            payload = json.loads(Path(sidecar).read_text(encoding='utf-8'))

        self.assertTrue(sidecar.endswith('_phase2h_r3_visual_review.json'))
        self.assertEqual(payload['phase'], '2H-R.3')
        self.assertEqual(payload['name'], 'visual_review_local_continuity_candidates')
        self.assertTrue(payload['read_only'])
        self.assertTrue(payload['seam_flags_unchanged'])
        self.assertTrue(payload['not_applied_to_mesh'])
        self.assertFalse(payload['probabilities_used'])
        self.assertTrue(payload['diagnostic_paths_are_labels_only'])
        self.assertFalse(payload['active_phase2j_allowed'])
        self.assertEqual(
            payload['source_simulation_sidecar'],
            'prediction_unified_local_continuity_simulation_phase2h_r.json',
        )
        self.assertTrue(payload['compact_sidecar'])
        self.assertEqual([edge.use_seam for edge in mesh.edges], flags_before)
        self.assertIn('write_phase2k_r_tangent_audit_rescue', read_addon_file('operators.py'))
        self.assertIn('normalized_decision_summary', payload)
        self.assertIn('compactness_summary', payload)
        self.assertIn('write_phase2h_r3_visual_review', read_addon_file('operators.py'))

    def test_phase2hr3_reports_low_straightness_selected_candidates(self):
        seam_mapping = load_module('uvsp_phase2hr3_low_straightness_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(99, 100), (102, 103), (100, 101), (101, 102)],
            vertex_count=10000,
            coords={
                99: (-1.0, 0.0, 0.0),
                100: (0.0, 0.0, 0.0),
                101: (1.0, 0.0, 0.0),
                102: (1.2, 0.98, 0.0),
                103: (1.4, 1.96, 0.0),
                9999: (10.0, 10.0, 10.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )

        simulation = seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)
        payload = seam_mapping.build_phase2h_r3_visual_review(simulation)
        reports = payload['low_straightness_selected_candidates']

        self.assertTrue(reports)
        self.assertLess(reports[0]['path_straightness'], 0.50)
        self.assertTrue(reports[0]['low_straightness_warning'])
        self.assertTrue(reports[0]['severe_low_straightness_warning'])
        self.assertTrue(reports[0]['active_repair_blocked'])
        self.assertIn('conservative_length2_only', reports[0]['selected_by_policies'])
        self.assertFalse(payload['active_phase2j_allowed'])
        self.assertGreater(
            payload['normalized_decision_summary']['low_straightness_selected_count'],
            0,
        )
        self.assertGreater(
            payload['normalized_decision_summary']['severe_low_straightness_selected_count'],
            0,
        )

    def test_phase2hr3_reports_straight_but_tangent_weak_audit_candidates(self):
        seam_mapping = load_module('uvsp_phase2hr3_tangent_weak_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(99, 100), (102, 103), (100, 101), (101, 102)],
            vertex_count=10000,
            coords={
                99: (0.0, -1.0, 0.0),
                100: (0.0, 0.0, 0.0),
                101: (1.0, 0.0, 0.0),
                102: (2.0, 0.0, 0.0),
                103: (3.0, 0.0, 0.0),
                9999: (10.0, 10.0, 10.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.build_phase2h_r3_visual_review(
            seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)
        )
        reports = payload['straight_but_tangent_weak_candidates']

        self.assertTrue(reports)
        self.assertGreaterEqual(reports[0]['path_straightness'], 0.85)
        self.assertLess(reports[0]['min_endpoint_tangent_alignment'], 0.30)
        self.assertTrue(reports[0]['tangent_model_audit_needed'])
        self.assertTrue(reports[0]['do_not_select_automatically'])
        self.assertTrue(reports[0]['active_repair_blocked'])
        self.assertEqual(
            reports[0]['recommended_architect_action'],
            'audit_tangent_estimator_do_not_auto_select',
        )

    def test_phase2hr3_separates_high_risk_probe_candidates(self):
        seam_mapping = load_module('uvsp_phase2hr3_high_risk_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(0, 1), (3, 4), (1, 2), (2, 3)],
            vertex_count=5,
            coords={
                0: (0.0, 0.0, 0.0),
                1: (0.01, 0.0, 0.0),
                2: (0.02, 0.0, 0.0),
                3: (0.03, 0.0, 0.0),
                4: (0.04, 0.0, 0.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(0, 1), (3, 4)],
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.build_phase2h_r3_visual_review(
            seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)
        )
        reports = payload['high_risk_probe_selected_candidates']

        self.assertTrue(reports)
        self.assertIn(reports[0]['candidate_class'], {
            'one_edge_endpoint_to_skeleton',
            'two_edge_endpoint_to_skeleton_or_near_junction',
            'three_edge_local_bridge',
            'two_edge_same_component_local_closure',
            'two_edge_tangent_failed_endpoint_bridge',
            'non_original_or_missing_blender_edge',
        })
        self.assertNotEqual(reports[0]['candidate_class'], 'two_edge_inter_component_endpoint_bridge')
        self.assertIn('class_balanced_probe', reports[0]['selected_by_policies'])
        self.assertTrue(reports[0]['would_require_new_repair_class'])
        self.assertTrue(reports[0]['active_repair_blocked'])
        self.assertGreater(
            payload['normalized_decision_summary']['high_risk_probe_selected_count'],
            0,
        )

    def test_phase2hr3_reports_rejected_tangent_failed_residuals(self):
        seam_mapping = load_module('uvsp_phase2hr3_tangent_failed_residual_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(3097, 3098), (3192, 3193), (3098, 3185), (3185, 3192)],
            vertex_count=4000,
            coords={
                3097: (0.0, -1.0, 0.0),
                3098: (0.0, 0.0, 0.0),
                3185: (1.0, 0.0, 0.0),
                3192: (2.0, 0.0, 0.0),
                3193: (3.0, 0.0, 0.0),
                9999: (10.0, 10.0, 10.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(3097, 3098), (3192, 3193)],
            clear_existing=True,
            enable_local_repair=False,
        )
        residual_payload = {
            'paths': [
                {
                    'label': '3a',
                    'path_vertex_ids': [3098, 3185, 3192],
                    'candidate_class_phase2e': 'phase_2b1_tangent_failed',
                    'phase_2b1_rejection_reason': 'tangent_alignment_failed',
                },
            ],
            'read_only': True,
        }

        payload = seam_mapping.build_phase2h_r3_visual_review(
            seam_mapping.simulate_unified_local_continuity_phase2h_r(
                mesh,
                residual_payload=residual_payload,
            )
        )
        reports = payload['rejected_tangent_failed_residuals']

        self.assertTrue(reports)
        self.assertIn('3a', reports[0]['residual_labels'])
        self.assertIn(reports[0]['reason_not_selected'], {
            'not_selected_by_simulation_policy',
            'straight_but_tangent_weak_do_not_select_automatically',
        })
        self.assertTrue(reports[0]['active_repair_blocked'])
        self.assertGreater(
            payload['normalized_decision_summary']['rejected_tangent_failed_residual_count'],
            0,
        )

    def test_phase2hr3_diagnostic_labels_do_not_affect_marking_or_selection(self):
        seam_mapping = load_module('uvsp_phase2hr3_labels_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_endpoint_bridge_mesh()
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        flags_before = [edge.use_seam for edge in mesh.edges]
        no_label = seam_mapping.build_phase2h_r3_visual_review(
            seam_mapping.simulate_unified_local_continuity_phase2h_r(mesh)
        )
        with_label = seam_mapping.build_phase2h_r3_visual_review(
            seam_mapping.simulate_unified_local_continuity_phase2h_r(
                mesh,
                residual_payload={
                    'paths': [
                        {
                            'label': 'manual_label',
                            'path_vertex_ids': [100, 101, 102],
                            'candidate_class_phase2e': 'manual',
                        },
                    ],
                    'read_only': True,
                },
            )
        )

        self.assertEqual([edge.use_seam for edge in mesh.edges], flags_before)
        self.assertEqual(
            no_label['compactness_summary']['counts_by_review_group'],
            with_label['compactness_summary']['counts_by_review_group'],
        )
        self.assertFalse(with_label['probabilities_used'])
        self.assertFalse(with_label['active_phase2j_allowed'])

    def test_phase2jr_small_gap_sidecar_is_emitted_and_read_only(self):
        seam_mapping = load_module('uvsp_phase2jr_sidecar_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_endpoint_bridge_mesh()
        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
            collect_debug_diagnostics=True,
        )
        flags_before = [edge.use_seam for edge in mesh.edges]

        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path = str(Path(temp_dir) / 'prediction.json')
            Path(prediction_path).write_text('{}', encoding='utf-8')
            sidecar = seam_mapping.write_phase2j_r_small_gap_rule_simulation(prediction_path, result)
            payload = json.loads(Path(sidecar).read_text(encoding='utf-8'))

        self.assertTrue(sidecar.endswith('_phase2j_r_small_gap_rule_simulation.json'))
        self.assertEqual(payload['phase'], '2J-R')
        self.assertEqual(payload['name'], 'small_local_gap_closure_rule_simulation_v2')
        self.assertTrue(payload['read_only'])
        self.assertTrue(payload['seam_flags_unchanged'])
        self.assertTrue(payload['not_applied_to_mesh'])
        self.assertFalse(payload['probabilities_used'])
        self.assertTrue(payload['diagnostic_paths_are_labels_only'])
        self.assertTrue(payload['active_phase2j_allowed_means_review_only'])
        self.assertTrue(payload['compact_sidecar'])
        self.assertEqual([edge.use_seam for edge in mesh.edges], flags_before)
        self.assertIn('decision_summary', payload)
        self.assertIn('compactness_summary', payload)
        self.assertIn('write_phase2j_r_small_gap_rule_simulation', read_addon_file('operators.py'))

    def test_phase2jr_curved_candidates_require_strict_guards(self):
        seam_mapping = load_module('uvsp_phase2jr_curved_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(99, 100), (102, 103), (100, 101), (101, 102)],
            vertex_count=10000,
            coords={
                99: (-1.0, 0.0, 0.0),
                100: (0.0, 0.0, 0.0),
                101: (1.0, 0.0, 0.0),
                102: (1.2, 0.98, 0.0),
                103: (1.4, 1.96, 0.0),
                9999: (1000.0, 1000.0, 1000.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_phase2j_r_small_gap_rule(
            mesh,
            residual_payload={
                'paths': [{'label': 'curved_case', 'path_vertex_ids': [100, 101, 102]}],
                'read_only': True,
            },
        )
        reports = payload['curved_two_edge_endpoint_bridge_candidates']

        self.assertTrue(reports)
        self.assertEqual(reports[0]['path_vertex_ids'], [100, 101, 102])
        self.assertEqual(reports[0]['degree_pattern'], [1, 0, 1])
        self.assertEqual(reports[0]['component_relation'], 'different_components')
        self.assertEqual(reports[0]['loop_risk'], 'none')
        self.assertEqual(reports[0]['topology_risk'], 'accepted_pattern')
        self.assertLessEqual(reports[0]['normalized_total_path_length'], 0.015)
        self.assertGreaterEqual(reports[0]['min_endpoint_tangent_alignment'], 0.75)
        self.assertLess(reports[0]['path_straightness'], 0.50)
        self.assertTrue(reports[0]['strict_rule_passed'])
        self.assertTrue(reports[0]['would_be_active_candidate_under_strict_rule'])
        self.assertTrue(reports[0]['visual_confirmation_required'])
        self.assertFalse(payload['probabilities_used'])

    def test_phase2jr_straight_tangent_weak_requires_audit_support(self):
        seam_mapping = load_module('uvsp_phase2jr_tangent_audit_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[(99, 100), (102, 103), (100, 101), (101, 102)],
            vertex_count=10000,
            coords={
                99: (0.0, -1.0, 0.0),
                100: (0.0, 0.0, 0.0),
                101: (1.0, 0.0, 0.0),
                102: (2.0, 0.0, 0.0),
                103: (3.0, 0.0, 0.0),
                9999: (1000.0, 1000.0, 1000.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_phase2j_r_small_gap_rule(mesh)
        reports = payload['straight_but_tangent_weak_audit_candidates']

        self.assertTrue(reports)
        self.assertEqual(reports[0]['path_vertex_ids'], [100, 101, 102])
        self.assertGreaterEqual(reports[0]['path_straightness'], 0.90)
        self.assertLess(reports[0]['min_endpoint_tangent_alignment'], 0.30)
        self.assertEqual(reports[0]['tangent_audit_status'], 'audit_supported_rescue')
        self.assertTrue(reports[0]['would_be_active_candidate_under_strict_rule'])
        self.assertTrue(reports[0]['visual_confirmation_required'])
        self.assertTrue(payload['active_phase2j_allowed_means_review_only'])

    def test_phase2jr_length_three_and_same_component_are_diagnostic_only(self):
        seam_mapping = load_module('uvsp_phase2jr_diagnostic_classes_smoke', ADDON_DIR / 'seam_mapping.py')
        length3_mesh = FakeMesh(
            edges=[(0, 1), (4, 5), (1, 2), (2, 3), (3, 4)],
            vertex_count=10000,
            coords={
                0: (0.0, 0.0, 0.0),
                1: (0.01, 0.0, 0.0),
                2: (0.02, 0.0, 0.0),
                3: (0.03, 0.0, 0.0),
                4: (0.04, 0.0, 0.0),
                5: (0.05, 0.0, 0.0),
                9999: (1.0, 1.0, 1.0),
            },
        )
        seam_mapping.apply_seam_keys(
            length3_mesh,
            [(0, 1), (4, 5)],
            clear_existing=True,
            enable_local_repair=False,
        )
        length3_payload = seam_mapping.simulate_phase2j_r_small_gap_rule(length3_mesh)
        length3_reports = length3_payload['length_three_local_bridge_diagnostics']

        same_component_mesh, predicted_keys, _ = build_endpoint_bridge_mesh(same_component=True)
        seam_mapping.apply_seam_keys(
            same_component_mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        same_component_payload = seam_mapping.simulate_phase2j_r_small_gap_rule(same_component_mesh)
        same_component_reports = same_component_payload['same_component_local_closure_diagnostics']

        self.assertTrue(length3_reports)
        self.assertFalse(length3_reports[0]['length3_active_safe'])
        self.assertIn('length3_active_repair_not_implemented_in_phase2j_r', length3_reports[0]['blocking_reasons'])
        self.assertTrue(same_component_reports)
        self.assertFalse(same_component_reports[0]['same_component_active_safe'])
        self.assertIn(
            'same_component_active_repair_not_implemented_in_phase2j_r',
            same_component_reports[0]['blocking_reasons'],
        )

    def test_phase2jr_representative_missing_edges_are_never_active_safe(self):
        seam_mapping = load_module('uvsp_phase2jr_missing_edges_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_endpoint_bridge_mesh()
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_phase2j_r_small_gap_rule(
            mesh,
            residual_payload={
                'paths': [{'label': 'missing_case', 'path_vertex_ids': [234, 319, 318, 214]}],
                'read_only': True,
            },
        )
        row = next(
            item for item in payload['representative_case_matrix']
            if item['path_vertex_ids'] == [234, 319, 318, 214]
        )

        self.assertTrue(row['found_in_candidate_space'])
        self.assertFalse(row['all_edges_exist_in_blender'])
        self.assertFalse(row['active_safe_under_strict_rule'])
        self.assertIn('missing_or_non_original_edges_not_active_safe', row['blocking_reasons'])

    def test_phase2jr_labels_do_not_affect_selection_or_marking(self):
        seam_mapping = load_module('uvsp_phase2jr_labels_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_endpoint_bridge_mesh()
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        flags_before = [edge.use_seam for edge in mesh.edges]
        no_label = seam_mapping.simulate_phase2j_r_small_gap_rule(mesh)
        with_label = seam_mapping.simulate_phase2j_r_small_gap_rule(
            mesh,
            residual_payload={
                'paths': [{'label': 'manual_label', 'path_vertex_ids': [100, 101, 102]}],
                'read_only': True,
            },
        )

        self.assertEqual([edge.use_seam for edge in mesh.edges], flags_before)
        self.assertEqual(
            no_label['curved_two_edge_endpoint_bridge_candidates'],
            with_label['curved_two_edge_endpoint_bridge_candidates'],
        )
        self.assertEqual(
            no_label['straight_but_tangent_weak_audit_candidates'],
            with_label['straight_but_tangent_weak_audit_candidates'],
        )
        self.assertFalse(with_label['probabilities_used'])

    def test_phase2jr_keeps_existing_phase2_constraints_intact(self):
        seam_mapping = load_module('uvsp_phase2jr_constraints_smoke', ADDON_DIR / 'seam_mapping.py')

        self.assertNotIn('target_paths', inspect.signature(seam_mapping.apply_two_edge_local_continuity_repair).parameters)

    def test_phase2j_curved_endpoint_bridge_marks_generic_candidate(self):
        seam_mapping = load_module('uvsp_phase2j_curved_mark_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, paths = build_curved_endpoint_bridge_mesh()
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        repair = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(mesh)
        selected = repair['selected_reports']

        self.assertEqual(repair['eligible_total'], 1)
        self.assertEqual(repair['paths_marked'], 1)
        self.assertEqual(repair['edges_marked'], 2)
        self.assertEqual(selected[0]['path_vertex_ids'], list(paths[0]))
        self.assertTrue(selected[0]['marked'])
        self.assertTrue(selected[0]['selected_for_marking'])
        self.assertIsNone(selected[0]['rejection_reason'])
        for edge_index in selected[0]['blender_edge_indices']:
            self.assertTrue(mesh.edges[edge_index].use_seam)
        self.assertFalse(selected[0]['diagnostic_labels_used_for_selection'])
        self.assertFalse(selected[0]['probabilities_used_for_selection'])

    def test_phase2j_curved_endpoint_bridge_rejects_unsafe_classes(self):
        seam_mapping = load_module('uvsp_phase2j_curved_reject_classes_smoke', ADDON_DIR / 'seam_mapping.py')
        same_component_mesh, predicted_keys, _ = build_endpoint_bridge_mesh(same_component=True)
        seam_mapping.apply_seam_keys(
            same_component_mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        same_component = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(same_component_mesh)

        length3_mesh = FakeMesh(
            edges=[(0, 1), (4, 5), (1, 2), (2, 3), (3, 4)],
            vertex_count=10000,
            coords={
                0: (0.0, 0.0, 0.0),
                1: (0.01, 0.0, 0.0),
                2: (0.02, 0.0, 0.0),
                3: (0.03, 0.0, 0.0),
                4: (0.04, 0.0, 0.0),
                5: (0.05, 0.0, 0.0),
                9999: (1000.0, 1000.0, 1000.0),
            },
        )
        seam_mapping.apply_seam_keys(
            length3_mesh,
            [(0, 1), (4, 5)],
            clear_existing=True,
            enable_local_repair=False,
        )
        length3 = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(length3_mesh)

        missing_mesh = FakeMesh(
            edges=[(0, 1), (3, 4)],
            vertex_count=5,
            coords={
                0: (0.0, 0.0, 0.0),
                1: (0.01, 0.0, 0.0),
                3: (0.03, 0.0, 0.0),
                4: (0.04, 0.0, 0.0),
            },
        )
        seam_mapping.apply_seam_keys(
            missing_mesh,
            [(0, 1), (3, 4)],
            clear_existing=True,
            enable_local_repair=False,
        )
        missing = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(missing_mesh)

        self.assertEqual(same_component['paths_marked'], 0)
        self.assertTrue(any(
            report['rejection_reason'] == 'same_component_not_endpoint_bridge'
            for report in same_component['candidate_reports']
        ))
        self.assertEqual(length3['paths_marked'], 0)
        self.assertFalse(any(len(report['path_vertex_ids']) == 4 for report in length3['candidate_reports']))
        self.assertEqual(missing['paths_marked'], 0)

    def test_phase2j_curved_endpoint_bridge_rejects_metric_failures_and_already_seam_edges(self):
        seam_mapping = load_module('uvsp_phase2j_curved_metric_reject_smoke', ADDON_DIR / 'seam_mapping.py')
        weak_tangent_mesh = FakeMesh(
            edges=[(99, 100), (102, 103), (100, 101), (101, 102)],
            vertex_count=10000,
            coords={
                99: (-1.0, 0.0, 0.0),
                100: (0.0, 0.0, 0.0),
                101: (1.0, 0.0, 0.0),
                102: (1.2, 0.98, 0.0),
                103: (2.2, 0.98, 0.0),
                9999: (1000.0, 1000.0, 1000.0),
            },
        )
        seam_mapping.apply_seam_keys(
            weak_tangent_mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )
        weak_tangent = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(weak_tangent_mesh)

        long_mesh, predicted_keys, _ = build_curved_endpoint_bridge_mesh()
        long_mesh.vertices[9999].co = (1.3, 1.0, 0.0)
        seam_mapping.apply_seam_keys(
            long_mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        too_long = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(long_mesh)

        low_ratio_mesh = FakeMesh(
            edges=[(99, 100), (102, 103), (100, 101), (101, 102)],
            vertex_count=10000,
            coords={
                99: (-1.0, 0.0, 0.0),
                100: (0.0, 0.0, 0.0),
                101: (1.0, 0.0, 0.0),
                102: (0.2, 0.6, 0.0),
                103: (-0.6, 1.2, 0.0),
                9999: (1000.0, 1000.0, 1000.0),
            },
        )
        seam_mapping.apply_seam_keys(
            low_ratio_mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )
        low_ratio = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(low_ratio_mesh)

        already_mesh, predicted_keys, _ = build_curved_endpoint_bridge_mesh()
        predicted_keys = list(predicted_keys) + [(100, 101)]
        seam_mapping.apply_seam_keys(
            already_mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        already = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(already_mesh)

        self.assertEqual(weak_tangent['paths_marked'], 0)
        self.assertTrue(any(
            report['rejection_reason'] == 'tangent_alignment_below_threshold'
            for report in weak_tangent['candidate_reports']
        ))
        self.assertEqual(too_long['paths_marked'], 0)
        self.assertTrue(any(
            report['rejection_reason'] == 'path_too_long'
            for report in too_long['candidate_reports']
        ))
        self.assertEqual(low_ratio['paths_marked'], 0)
        self.assertTrue(any(
            report['rejection_reason'] == 'segment_length_ratio_below_threshold'
            for report in low_ratio['candidate_reports']
        ))
        self.assertEqual(already['paths_marked'], 0)
        self.assertTrue(any(
            report['rejection_reason'] == 'edge_already_seam'
            for report in already['candidate_reports']
        ))

    def test_phase2j_curved_endpoint_bridge_duplicate_endpoint_pair_keeps_best_ranked(self):
        seam_mapping = load_module('uvsp_phase2j_curved_duplicate_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (99, 100),
                (102, 103),
                (100, 101),
                (101, 102),
                (100, 104),
                (104, 102),
            ],
            vertex_count=10000,
            coords={
                99: (-1.0, 0.0, 0.0),
                100: (0.0, 0.0, 0.0),
                101: (1.0, 0.0, 0.0),
                102: (1.2, 0.98, 0.0),
                103: (1.4, 1.96, 0.0),
                104: (0.9, 0.1, 0.0),
                9999: (1000.0, 1000.0, 1000.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )

        repair = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(mesh)

        self.assertEqual(repair['eligible_total'], 1)
        self.assertEqual(repair['paths_marked'], 1)
        self.assertEqual(repair['selected_reports'][0]['path_vertex_ids'], [100, 101, 102])
        self.assertTrue(any(
            report['path_vertex_ids'] == [100, 104, 102]
            and report['rejection_reason'] == 'duplicate_endpoint_pair_suppressed'
            for report in repair['candidate_reports']
        ))

    def test_phase2j_curved_endpoint_bridge_shared_intermediate_vertex_does_not_suppress(self):
        seam_mapping = load_module('uvsp_phase2j_curved_shared_middle_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh = FakeMesh(
            edges=[
                (90, 100),
                (102, 112),
                (100, 101),
                (101, 102),
                (210, 200),
                (202, 212),
                (200, 101),
                (101, 202),
            ],
            vertex_count=10000,
            coords={
                90: (-2.0, 0.0, 0.0),
                100: (-1.0, 0.0, 0.0),
                101: (0.0, 0.0, 0.0),
                102: (-0.2, 0.98, 0.0),
                112: (-0.4, 1.96, 0.0),
                210: (2.0, 0.0, 0.0),
                200: (1.0, 0.0, 0.0),
                202: (0.2, -0.98, 0.0),
                212: (0.4, -1.96, 0.0),
                9999: (1000.0, 1000.0, 1000.0),
            },
        )
        seam_mapping.apply_seam_keys(
            mesh,
            [(90, 100), (102, 112), (210, 200), (202, 212)],
            clear_existing=True,
            enable_local_repair=False,
        )

        repair = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(mesh)
        selected_paths = {tuple(report['path_vertex_ids']) for report in repair['selected_reports']}

        self.assertEqual(repair['paths_marked'], 2)
        self.assertEqual({path[1] for path in selected_paths}, {101})
        self.assertEqual(len({(path[0], path[2]) for path in selected_paths}), 2)
        self.assertFalse(any(
            report['rejection_reason'] == 'duplicate_endpoint_pair_suppressed'
            for report in repair['candidate_reports']
        ))

    def test_phase2j_curved_endpoint_bridge_cap_and_ranking_are_deterministic(self):
        seam_mapping = load_module('uvsp_phase2j_curved_cap_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, paths = build_curved_endpoint_bridge_mesh(count=8)
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        repair = seam_mapping.apply_curved_two_edge_endpoint_bridge_repair(mesh)
        selected_paths = [tuple(report['path_vertex_ids']) for report in repair['selected_reports']]

        self.assertTrue(repair['over_cap'])
        self.assertEqual(repair['eligible_total'], 8)
        self.assertEqual(repair['paths_marked'], 6)
        self.assertEqual(repair['edges_marked'], 12)
        self.assertEqual(selected_paths, [tuple(path) for path in paths[:6]])
        self.assertTrue(any(
            report['rejection_reason'] == 'repair_over_cap'
            for report in repair['candidate_reports']
        ))

    def test_phase2j_curved_endpoint_bridge_diagnostic_labels_and_probabilities_are_not_used(self):
        seam_mapping = load_module('uvsp_phase2j_curved_integrity_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_curved_endpoint_bridge_mesh()
        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
        )

        source = inspect.getsource(seam_mapping.apply_curved_two_edge_endpoint_bridge_repair)
        source += inspect.getsource(seam_mapping._curved_two_edge_endpoint_bridge_report)
        self.assertNotIn('DIAGNOSTIC_HUMAN_GAP_REGRESSION_PATHS', source)
        self.assertNotIn('DIAGNOSTIC_RESIDUAL_GAP_PHASE2E_PATHS', source)
        self.assertNotIn('DIAGNOSTIC_OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS', source)
        self.assertFalse(result.probabilities_used_for_curved_repair)
        self.assertFalse(result.phase2j_curved_repair_uses_hardcoded_paths)
        self.assertFalse(result.phase2j_curved_repair_uses_probabilities)
        self.assertTrue(all(
            report['diagnostic_labels_used_for_selection'] is False
            for report in result.blender_curved_two_edge_endpoint_bridge_candidate_reports
        ))

    def test_phase2j_curved_endpoint_bridge_keeps_existing_phase2_constraints_intact(self):
        seam_mapping = load_module('uvsp_phase2j_curved_constraints_smoke', ADDON_DIR / 'seam_mapping.py')

        self.assertEqual(inspect.signature(seam_mapping.apply_curved_two_edge_endpoint_bridge_repair).parameters[
            'max_repair_paths'
        ].default, 6)
        self.assertNotIn('target_paths', inspect.signature(seam_mapping.apply_two_edge_local_continuity_repair).parameters)

    def test_phase2k_r_tangent_audit_sidecar_is_emitted_and_read_only(self):
        seam_mapping = load_module('uvsp_phase2kr_sidecar_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_straight_tangent_weak_mesh(alternative_support=True)
        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        flags_before = [edge.use_seam for edge in mesh.edges]

        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path = str(Path(temp_dir) / 'prediction.json')
            Path(prediction_path).write_text('{}', encoding='utf-8')
            sidecar = seam_mapping.write_phase2k_r_tangent_audit_rescue(prediction_path, result)
            payload = json.loads(Path(sidecar).read_text(encoding='utf-8'))

        self.assertTrue(sidecar.endswith('_phase2k_r_tangent_audit_rescue.json'))
        self.assertEqual(payload['phase'], '2K-R')
        self.assertEqual(payload['name'], 'straight_tangent_weak_rescue_audit')
        self.assertTrue(payload['read_only'])
        self.assertTrue(payload['seam_flags_unchanged'])
        self.assertTrue(payload['not_applied_to_mesh'])
        self.assertFalse(payload['probabilities_used'])
        self.assertTrue(payload['diagnostic_paths_are_labels_only'])
        self.assertTrue(payload['active_phase2k_allowed_means_review_only'])
        self.assertTrue(payload['compact_sidecar'])
        self.assertEqual([edge.use_seam for edge in mesh.edges], flags_before)

    def test_phase2k_r_reports_straight_one_sided_weak_candidates_without_audit_evidence(self):
        seam_mapping = load_module('uvsp_phase2kr_inconclusive_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path = build_straight_tangent_weak_mesh()
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_phase2k_r_tangent_audit_rescue(mesh)
        reports = payload['tangent_audit_candidates']

        self.assertTrue(reports)
        self.assertEqual(reports[0]['path_vertex_ids'], list(path))
        self.assertEqual(reports[0]['tangent_audit_status'], 'audit_inconclusive')
        self.assertFalse(reports[0]['active_safe_under_future_phase2k'])
        self.assertEqual(reports[0]['rejection_or_blocking_reason'], 'no_explicit_tangent_audit_support')
        self.assertFalse(payload['active_phase2k_allowed'])

    def test_phase2k_r_active_safe_requires_explicit_alternative_tangent_support(self):
        seam_mapping = load_module('uvsp_phase2kr_supported_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path = build_straight_tangent_weak_mesh(alternative_support=True)
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_phase2k_r_tangent_audit_rescue(mesh)
        reports = payload['tangent_audit_candidates']

        self.assertTrue(reports)
        self.assertEqual(reports[0]['path_vertex_ids'], list(path))
        self.assertEqual(reports[0]['tangent_audit_status'], 'audit_supported_rescue')
        self.assertTrue(reports[0]['explicit_audit_support']['alternative_tangent_supported'])
        self.assertTrue(reports[0]['active_safe_under_future_phase2k'])
        self.assertTrue(reports[0]['active_safe_under_future_phase2k_means_review_only'])
        self.assertTrue(payload['active_phase2k_allowed'])

    def test_phase2k_r_rejects_same_component_length3_and_missing_edges_from_audit_candidates(self):
        seam_mapping = load_module('uvsp_phase2kr_reject_scope_smoke', ADDON_DIR / 'seam_mapping.py')
        same_mesh, same_predicted, _ = build_straight_tangent_weak_mesh(
            alternative_support=True,
            same_component=True,
        )
        seam_mapping.apply_seam_keys(
            same_mesh,
            same_predicted,
            clear_existing=True,
            enable_local_repair=False,
        )
        same_payload = seam_mapping.simulate_phase2k_r_tangent_audit_rescue(same_mesh)

        length3_mesh = FakeMesh(
            edges=[(0, 1), (4, 5), (1, 2), (2, 3), (3, 4)],
            vertex_count=10000,
            coords={
                0: (0.0, 0.0, 0.0),
                1: (0.01, 0.0, 0.0),
                2: (0.02, 0.0, 0.0),
                3: (0.03, 0.0, 0.0),
                4: (0.04, 0.0, 0.0),
                5: (0.05, 0.0, 0.0),
                9999: (1000.0, 1000.0, 1000.0),
            },
        )
        seam_mapping.apply_seam_keys(
            length3_mesh,
            [(0, 1), (4, 5)],
            clear_existing=True,
            enable_local_repair=False,
        )
        length3_payload = seam_mapping.simulate_phase2k_r_tangent_audit_rescue(length3_mesh)

        missing_payload = seam_mapping.simulate_phase2k_r_tangent_audit_rescue(
            same_mesh,
            residual_payload={
                'paths': [{'label': 'missing_case', 'path_vertex_ids': [5477, 5520, 5483]}],
                'read_only': True,
            },
        )

        self.assertEqual(same_payload['decision_summary']['straight_tangent_weak_candidates_total'], 0)
        self.assertEqual(length3_payload['decision_summary']['straight_tangent_weak_candidates_total'], 0)
        self.assertFalse(missing_payload['representative_case']['all_edges_exist_in_blender'])
        self.assertFalse(missing_payload['representative_case']['tangent_audit_candidate'])

    def test_phase2k_r_diagnostic_labels_do_not_affect_active_safe_decision(self):
        seam_mapping = load_module('uvsp_phase2kr_labels_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path = build_straight_tangent_weak_mesh(alternative_support=True)
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        no_label = seam_mapping.simulate_phase2k_r_tangent_audit_rescue(mesh)
        with_label = seam_mapping.simulate_phase2k_r_tangent_audit_rescue(
            mesh,
            residual_payload={
                'paths': [{'label': 'manual_label', 'path_vertex_ids': list(path)}],
                'read_only': True,
            },
        )

        self.assertEqual(
            no_label['tangent_audit_candidates'][0]['active_safe_under_future_phase2k'],
            with_label['tangent_audit_candidates'][0]['active_safe_under_future_phase2k'],
        )
        self.assertIn('manual_label', with_label['tangent_audit_candidates'][0]['diagnostic_labels_if_any'])
        self.assertFalse(with_label['tangent_audit_candidates'][0]['diagnostic_labels_used_for_selection'])

    def test_phase2k_r_representative_5477_style_case_appears_by_generic_rule(self):
        seam_mapping = load_module('uvsp_phase2kr_representative_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_straight_tangent_weak_mesh(
            path=(5477, 5520, 5483),
            alternative_support=True,
        )
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        payload = seam_mapping.simulate_phase2k_r_tangent_audit_rescue(mesh)
        representative = payload['representative_case']

        self.assertTrue(representative['found_in_candidate_space'])
        self.assertTrue(representative['all_edges_exist_in_blender'])
        self.assertTrue(representative['tangent_audit_candidate'])
        self.assertEqual(representative['tangent_audit_status'], 'audit_supported_rescue')
        self.assertTrue(representative['active_safe_under_future_phase2k'])
        self.assertFalse(representative['diagnostic_labels_used_for_selection'])
        self.assertTrue(any(
            report['path_vertex_ids'] == [5477, 5520, 5483]
            for report in payload['tangent_audit_candidates']
        ))

    def test_phase2k_r_phase2j_compact_audit_and_existing_caps_remain_intact(self):
        seam_mapping = load_module('uvsp_phase2kr_phase2j_audit_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_curved_endpoint_bridge_mesh()
        result = seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=True,
            collect_debug_diagnostics=True,
        )
        audit = result.phase2k_r_tangent_audit_rescue['phase2j_curved_repair_compact_audit']

        self.assertEqual(audit['curved_safety_cap'], 6)
        self.assertEqual(audit['curved_paths_marked'], result.blender_curved_two_edge_endpoint_bridge_paths_marked)
        self.assertIn('integrity_flags', audit)
        self.assertFalse(audit['integrity_flags']['probabilities_used_for_curved_repair'])
        self.assertFalse(audit['integrity_flags']['phase2j_curved_repair_uses_hardcoded_paths'])

    def test_phase2k_tangent_audit_rescue_marks_generic_supported_candidate(self):
        seam_mapping = load_module('uvsp_phase2k_active_mark_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path = build_straight_tangent_weak_mesh(alternative_support=True)
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        repair = seam_mapping.apply_tangent_audit_endpoint_bridge_rescue(mesh)
        selected = repair['selected_reports']

        self.assertEqual(repair['candidates_total'], 1)
        self.assertEqual(repair['eligible_total'], 1)
        self.assertEqual(repair['paths_marked'], 1)
        self.assertEqual(repair['edges_marked'], 2)
        self.assertEqual(selected[0]['path_vertex_ids'], list(path))
        self.assertTrue(selected[0]['marked'])
        self.assertGreaterEqual(selected[0]['best_alternative_tangent_alignment'], 0.85)
        self.assertFalse(selected[0]['diagnostic_labels_used_for_selection'])
        for edge_index in selected[0]['blender_edge_indices']:
            self.assertTrue(mesh.edges[edge_index].use_seam)

    def test_phase2k_tangent_audit_rescue_rejects_without_alternative_support(self):
        seam_mapping = load_module('uvsp_phase2k_active_no_support_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, _ = build_straight_tangent_weak_mesh()
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        repair = seam_mapping.apply_tangent_audit_endpoint_bridge_rescue(mesh)

        self.assertEqual(repair['paths_marked'], 0)
        self.assertTrue(any(
            report['rejection_reason'] == 'alternative_tangent_unavailable'
            for report in repair['candidate_reports']
        ))

    def test_phase2k_tangent_audit_rescue_rejects_both_weak_and_low_straightness(self):
        seam_mapping = load_module('uvsp_phase2k_active_metric_reject_smoke', ADDON_DIR / 'seam_mapping.py')
        both_weak_mesh = FakeMesh(
            edges=[(99, 100), (102, 103), (98, 100), (100, 101), (101, 102)],
            vertex_count=10000,
            coords={
                98: (-1.0, 0.0, 0.0),
                99: (0.0, -1.0, 0.0),
                100: (0.0, 0.0, 0.0),
                101: (1.0, 0.0, 0.0),
                102: (2.0, 0.0, 0.0),
                103: (2.0, 1.0, 0.0),
                9999: (1000.0, 1000.0, 1000.0),
            },
        )
        seam_mapping.apply_seam_keys(
            both_weak_mesh,
            [(99, 100), (102, 103)],
            clear_existing=True,
            enable_local_repair=False,
        )
        both_weak = seam_mapping.apply_tangent_audit_endpoint_bridge_rescue(both_weak_mesh)

        low_straight_mesh, predicted_keys, _ = build_straight_tangent_weak_mesh(alternative_support=True)
        low_straight_mesh.vertices[101].co = (1.0, 0.4, 0.0)
        seam_mapping.apply_seam_keys(
            low_straight_mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        low_straight = seam_mapping.apply_tangent_audit_endpoint_bridge_rescue(low_straight_mesh)

        self.assertEqual(both_weak['paths_marked'], 0)
        self.assertTrue(any(
            report['rejection_reason'] == 'strong_endpoint_tangent_below_threshold'
            for report in both_weak['candidate_reports']
        ))
        self.assertEqual(low_straight['paths_marked'], 0)
        self.assertTrue(any(
            report['rejection_reason'] == 'path_not_straight_enough'
            for report in low_straight['candidate_reports']
        ))

    def test_phase2k_tangent_audit_rescue_rejects_too_long_same_component_and_non_scope_paths(self):
        seam_mapping = load_module('uvsp_phase2k_active_scope_reject_smoke', ADDON_DIR / 'seam_mapping.py')
        long_mesh, predicted_keys, _ = build_straight_tangent_weak_mesh(alternative_support=True)
        long_mesh.vertices[9999].co = (2.1, 0.0, 0.0)
        seam_mapping.apply_seam_keys(
            long_mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        too_long = seam_mapping.apply_tangent_audit_endpoint_bridge_rescue(long_mesh)

        same_mesh, same_predicted, _ = build_straight_tangent_weak_mesh(
            alternative_support=True,
            same_component=True,
        )
        seam_mapping.apply_seam_keys(
            same_mesh,
            same_predicted,
            clear_existing=True,
            enable_local_repair=False,
        )
        same_component = seam_mapping.apply_tangent_audit_endpoint_bridge_rescue(same_mesh)

        length3_mesh = FakeMesh(
            edges=[(0, 1), (4, 5), (1, 2), (2, 3), (3, 4)],
            vertex_count=10000,
            coords={
                0: (0.0, 0.0, 0.0),
                1: (0.01, 0.0, 0.0),
                2: (0.02, 0.0, 0.0),
                3: (0.03, 0.0, 0.0),
                4: (0.04, 0.0, 0.0),
                5: (0.05, 0.0, 0.0),
                9999: (1000.0, 1000.0, 1000.0),
            },
        )
        seam_mapping.apply_seam_keys(
            length3_mesh,
            [(0, 1), (4, 5)],
            clear_existing=True,
            enable_local_repair=False,
        )
        length3 = seam_mapping.apply_tangent_audit_endpoint_bridge_rescue(length3_mesh)

        missing_mesh = FakeMesh(edges=[(0, 1), (3, 4)], vertex_count=5)
        seam_mapping.apply_seam_keys(
            missing_mesh,
            [(0, 1), (3, 4)],
            clear_existing=True,
            enable_local_repair=False,
        )
        missing = seam_mapping.apply_tangent_audit_endpoint_bridge_rescue(missing_mesh)

        self.assertEqual(too_long['paths_marked'], 0)
        self.assertTrue(any(
            report['rejection_reason'] == 'path_too_long'
            for report in too_long['candidate_reports']
        ))
        self.assertEqual(same_component['paths_marked'], 0)
        self.assertTrue(any(
            report['rejection_reason'] == 'same_component_not_endpoint_bridge'
            for report in same_component['candidate_reports']
        ))
        self.assertEqual(length3['paths_marked'], 0)
        self.assertEqual(missing['paths_marked'], 0)

    def test_phase2k_tangent_audit_rescue_cap_and_ranking_are_deterministic(self):
        seam_mapping = load_module('uvsp_phase2k_active_cap_smoke', ADDON_DIR / 'seam_mapping.py')
        edges = []
        predicted_keys = []
        coords = {9999: (1000.0, 1000.0, 1000.0)}
        paths = []
        for index in range(3):
            base = 100 + index * 10
            mesh_part, predicted_part, path = build_straight_tangent_weak_mesh(
                path=(base, base + 1, base + 2),
                alternative_support=True,
            )
            index_offset = len(edges)
            edges.extend([tuple(edge.vertices) for edge in mesh_part.edges])
            predicted_keys.extend(predicted_part)
            coords.update({vertex: item.co for vertex, item in enumerate(mesh_part.vertices) if item.co is not None})
            paths.append(path)
        mesh = FakeMesh(edges=edges, vertex_count=10000, coords=coords)
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )

        repair = seam_mapping.apply_tangent_audit_endpoint_bridge_rescue(mesh)

        self.assertTrue(repair['over_cap'])
        self.assertEqual(repair['eligible_total'], 3)
        self.assertEqual(repair['paths_marked'], 1)
        self.assertEqual(repair['edges_marked'], 2)
        self.assertEqual(repair['selected_reports'][0]['path_vertex_ids'], list(paths[0]))
        self.assertTrue(any(
            report['rejection_reason'] == 'repair_over_cap'
            for report in repair['candidate_reports']
        ))

    def test_phase2k_tangent_audit_rescue_diagnostic_labels_and_probabilities_are_not_used(self):
        seam_mapping = load_module('uvsp_phase2k_active_integrity_smoke', ADDON_DIR / 'seam_mapping.py')
        mesh, predicted_keys, path = build_straight_tangent_weak_mesh(alternative_support=True)
        seam_mapping.apply_seam_keys(
            mesh,
            predicted_keys,
            clear_existing=True,
            enable_local_repair=False,
        )
        no_label = seam_mapping.apply_tangent_audit_endpoint_bridge_rescue(mesh)

        source = inspect.getsource(seam_mapping.apply_tangent_audit_endpoint_bridge_rescue)
        source += inspect.getsource(seam_mapping._phase2k_active_tangent_rescue_report)
        self.assertNotIn('DIAGNOSTIC_HUMAN_GAP_REGRESSION_PATHS', source)
        self.assertNotIn('DIAGNOSTIC_RESIDUAL_GAP_PHASE2E_PATHS', source)
        self.assertNotIn('DIAGNOSTIC_OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS', source)
        self.assertTrue(all(
            report['diagnostic_labels_used_for_selection'] is False
            for report in no_label['candidate_reports']
        ))

        mesh2, predicted_keys2, _ = build_straight_tangent_weak_mesh(alternative_support=True)
        result = seam_mapping.apply_seam_keys(
            mesh2,
            predicted_keys2,
            clear_existing=True,
            enable_local_repair=True,
        )
        self.assertFalse(result.phase2k_tangent_rescue_uses_hardcoded_paths)
        self.assertFalse(result.phase2k_tangent_rescue_uses_probabilities)
        self.assertFalse(result.probabilities_used_for_tangent_rescue)
        self.assertEqual(inspect.signature(seam_mapping.apply_tangent_audit_endpoint_bridge_rescue).parameters[
            'max_repair_paths'
        ].default, 1)
        self.assertEqual(inspect.signature(seam_mapping.apply_curved_two_edge_endpoint_bridge_repair).parameters[
            'max_repair_paths'
        ].default, 6)

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
