import json
import math
from dataclasses import dataclass


@dataclass
class SeamApplyResult:
    requested: int
    unique: int
    applied: int
    ignored_non_original: int
    duplicates_skipped: int
    accepted_bridge_edges_present_in_json: int = 0
    accepted_bridge_edges_applied: int = 0
    accepted_bridge_edges_ignored_non_original: int = 0
    accepted_bridge_apply_trace: tuple = ()
    editable_gap_fill_result: dict | None = None


def load_predicted_edge_keys(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError('Prediction output must be a JSON object.')
    if payload.get('status') != 'ok':
        raise ValueError('Prediction output status is not ok.')

    seam_edges = payload.get('seam_edges')
    if not isinstance(seam_edges, list):
        raise ValueError('Prediction output seam_edges must be a list.')

    keys = []
    for index, entry in enumerate(seam_edges):
        if not isinstance(entry, dict):
            raise ValueError(f'Prediction edge #{index} must be an object.')
        vertex_ids = entry.get('vertex_ids_0based')
        if not _is_vertex_pair(vertex_ids):
            raise ValueError(f'Prediction edge #{index} has invalid vertex_ids_0based.')
        v0, v1 = vertex_ids
        keys.append((min(v0, v1), max(v0, v1)))

    return keys


def load_accepted_bridge_edge_keys(json_path):
    return [
        tuple(entry['vertex_ids_0based'])
        for entry in load_accepted_bridge_debug_entries(json_path)
    ]


def load_accepted_bridge_debug_entries(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError('Prediction output must be a JSON object.')
    diagnostics = payload.get('diagnostics')
    if not isinstance(diagnostics, dict):
        return []
    postprocess = diagnostics.get('postprocess')
    if not isinstance(postprocess, dict):
        return []
    bridging = postprocess.get('bridging')
    if not isinstance(bridging, dict):
        return []

    keys_by_edge_id = {}
    for index, value in enumerate(bridging.get('accepted_bridge_edge_keys', [])):
        if not _is_vertex_pair(value):
            raise ValueError(f'Accepted bridge edge #{index} has invalid vertex pair.')
        v0, v1 = value
        edge_id = None
        edge_indices = bridging.get('accepted_bridge_edge_indices', [])
        if index < len(edge_indices) and type(edge_indices[index]) is int:
            edge_id = int(edge_indices[index])
        keys_by_edge_id[edge_id if edge_id is not None else index] = (min(v0, v1), max(v0, v1))

    final_presence_by_edge_id = {}
    for report in bridging.get('bridge_edge_ids_final_presence', []):
        if isinstance(report, dict) and type(report.get('edge_id')) is int:
            final_presence_by_edge_id[int(report['edge_id'])] = report

    path_report_by_edge_id = {}
    for path_id, report in enumerate(bridging.get('accepted_bridge_reports', [])):
        if not isinstance(report, dict):
            continue
        for edge_id in report.get('path_edge_ids', []):
            if type(edge_id) is int:
                path_report_by_edge_id[int(edge_id)] = (path_id, report)

    entries = []
    for fallback_index, (edge_id, key) in enumerate(sorted(keys_by_edge_id.items(), key=lambda item: item[0])):
        path_id, path_report = path_report_by_edge_id.get(edge_id, (None, {}))
        final_report = final_presence_by_edge_id.get(edge_id, {})
        entries.append({
            'canonical_edge_index': None if edge_id is None else int(edge_id),
            'vertex_ids_0based': [int(key[0]), int(key[1])],
            'bridge_path_id': path_id,
            'path_edge_count': path_report.get('path_edge_count', path_report.get('bridge_edge_count')),
            'same_component': bool(path_report.get('same_component', False)),
            'present_in_final_json': bool(
                final_report.get(
                    'in_output_seam_edges',
                    final_report.get('in_after_stage_c', True),
                )
            ),
        })
    return entries


def apply_editable_shortest_path_gap_fill(
    mesh,
    *,
    enabled=True,
    max_gap_hops=2,
    allow_same_component=False,
    min_same_component_loop_size=8,
):
    if isinstance(max_gap_hops, bool) or int(max_gap_hops) < 1:
        raise ValueError('max_gap_hops must be an integer greater than or equal to 1')
    if isinstance(min_same_component_loop_size, bool) or int(min_same_component_loop_size) < 1:
        raise ValueError('min_same_component_loop_size must be an integer greater than or equal to 1')

    max_hops = int(max_gap_hops)
    min_loop = int(min_same_component_loop_size)
    if not enabled:
        return _editable_gap_fill_empty_result(False, max_hops)

    edge_items, edge_by_key, adjacency = _mesh_edge_lookup(mesh)
    seam_degree, seam_adjacency = _seam_topology_from_mesh_edges(edge_items)
    endpoints = sorted(vertex for vertex, degree in seam_degree.items() if int(degree) == 1)
    existing_seam_targets = sorted(
        vertex for vertex, degree in seam_degree.items() if int(degree) >= 2
    )
    seam_vertices = {int(vertex) for vertex, degree in seam_degree.items() if int(degree) > 0}
    component_id_of = _seam_component_ids(seam_adjacency)

    counters = {
        'rejected_same_component': 0,
        'rejected_endpoint_to_existing_same_component': 0,
        'rejected_endpoint_same_component_too_short': 0,
        'rejected_endpoint_to_existing_same_component_too_short': 0,
        'rejected_same_component_no_existing_path': 0,
        'rejected_existing_seam_internal': 0,
        'rejected_internal_seam_vertex': 0,
        'rejected_no_path': 0,
        'rejected_junction_gap_no_high_degree_endpoint': 0,
        'rejected_junction_gap_same_component_too_short': 0,
        'rejected_junction_gap_internal_seam_vertex': 0,
        'rejected_junction_gap_existing_seam_edge': 0,
        'rejected_junction_gap_reserved_edge': 0,
    }
    candidates = []

    def add_candidate(start, target, kind):
        same_component = component_id_of.get(start) == component_id_of.get(target)
        is_junction_gap = kind == 'junction_gap_closure'
        is_endpoint_loop_candidate = (
            same_component
            and kind in ('endpoint_to_endpoint', 'endpoint_to_existing_seam_vertex')
        )
        if is_junction_gap:
            if max(int(seam_degree.get(start, 0)), int(seam_degree.get(target, 0))) < 3:
                counters['rejected_junction_gap_no_high_degree_endpoint'] += 1
                return

        paths = _bounded_editable_paths(adjacency, start, target, max_hops)
        if not paths:
            counters['rejected_no_path'] += 1
            return

        valid_paths = []
        for path in paths:
            path_edges = _path_edge_keys(path)
            if any(bool(edge_by_key[edge_key][1].use_seam) for edge_key in path_edges):
                counters['rejected_existing_seam_internal'] += 1
                if is_junction_gap:
                    counters['rejected_junction_gap_existing_seam_edge'] += 1
                continue
            if any(int(vertex) in seam_vertices for vertex in path[1:-1]):
                counters['rejected_internal_seam_vertex'] += 1
                if is_junction_gap:
                    counters['rejected_junction_gap_internal_seam_vertex'] += 1
                continue
            if same_component:
                seam_distance = _shortest_seam_path_length(seam_adjacency, start, target)
                if is_junction_gap:
                    min_existing_seam_distance = max(6, max_hops + 3)
                    if seam_distance is None or seam_distance < min_existing_seam_distance:
                        counters['rejected_junction_gap_same_component_too_short'] += 1
                        continue
                else:
                    min_existing_seam_distance = max(6, max_hops + 3)
                    if seam_distance is None:
                        counters['rejected_same_component'] += 1
                        counters['rejected_same_component_no_existing_path'] += 1
                        if kind == 'endpoint_to_existing_seam_vertex':
                            counters['rejected_endpoint_to_existing_same_component'] += 1
                        continue
                    if seam_distance < min_existing_seam_distance:
                        counters['rejected_same_component'] += 1
                        if kind == 'endpoint_to_endpoint':
                            counters['rejected_endpoint_same_component_too_short'] += 1
                        if kind == 'endpoint_to_existing_seam_vertex':
                            counters['rejected_endpoint_to_existing_same_component'] += 1
                            counters['rejected_endpoint_to_existing_same_component_too_short'] += 1
                        continue
            valid_paths.append(path)

        if not valid_paths:
            return
        best_path = min(
            valid_paths,
            key=lambda path: (
                len(path) - 1,
                _editable_path_length(mesh, path),
                tuple(path),
            ),
        )
        candidates.append(_editable_gap_candidate(
            mesh,
            best_path,
            kind,
            start,
            target,
            same_component_loop_closure=bool(is_endpoint_loop_candidate),
        ))

    for left_index, left in enumerate(endpoints):
        for right in endpoints[left_index + 1:]:
            add_candidate(left, right, 'endpoint_to_endpoint')

    for start in endpoints:
        for target in existing_seam_targets:
            add_candidate(start, target, 'endpoint_to_existing_seam_vertex')

    junction_gap_vertices = sorted(
        vertex for vertex, degree in seam_degree.items()
        if int(degree) >= 2
    )
    for left_index, left in enumerate(junction_gap_vertices):
        for right in junction_gap_vertices[left_index + 1:]:
            add_candidate(left, right, 'junction_gap_closure')

    kind_priority = {
        'endpoint_to_endpoint': 0,
        'endpoint_to_existing_seam_vertex': 1,
        'junction_gap_closure': 2,
    }
    candidates.sort(
        key=lambda item: (
            item['hop_count'],
            item['total_length'],
            kind_priority.get(item['kind'], 99),
            item['start_vertex'],
            item['target_vertex'],
            tuple(item['vertices']),
        )
    )

    consumed_endpoints = set()
    reserved_edges = set()
    accepted_paths = []
    accepted_edge_keys = set()
    rejected_conflict_consumed_endpoint = 0
    for candidate in candidates:
        consumed_candidate_endpoints = tuple(candidate['consumed_endpoint_vertices'])
        edge_keys = {tuple(edge_key) for edge_key in candidate['edges']}
        if any(vertex in consumed_endpoints for vertex in consumed_candidate_endpoints):
            rejected_conflict_consumed_endpoint += 1
            continue
        if edge_keys & reserved_edges:
            if candidate['kind'] == 'junction_gap_closure':
                counters['rejected_junction_gap_reserved_edge'] += 1
            else:
                rejected_conflict_consumed_endpoint += 1
            continue
        for edge_key in edge_keys:
            edge_by_key[edge_key][1].use_seam = True
            accepted_edge_keys.add(edge_key)
        consumed_endpoints.update(consumed_candidate_endpoints)
        reserved_edges.update(edge_keys)
        accepted_paths.append(candidate)

    endpoint_to_existing_candidates = sum(
        1 for candidate in candidates
        if candidate['kind'] == 'endpoint_to_existing_seam_vertex'
    )
    endpoint_to_existing_accepted = sum(
        1 for candidate in accepted_paths
        if candidate['kind'] == 'endpoint_to_existing_seam_vertex'
    )
    endpoint_loop_closure_candidates = sum(
        1 for candidate in candidates
        if (
            candidate['kind'] == 'endpoint_to_endpoint'
            and candidate.get('same_component_loop_closure')
        )
    )
    endpoint_loop_closure_accepted = sum(
        1 for candidate in accepted_paths
        if (
            candidate['kind'] == 'endpoint_to_endpoint'
            and candidate.get('same_component_loop_closure')
        )
    )
    endpoint_to_existing_loop_closure_candidates = sum(
        1 for candidate in candidates
        if (
            candidate['kind'] == 'endpoint_to_existing_seam_vertex'
            and candidate.get('same_component_loop_closure')
        )
    )
    endpoint_to_existing_loop_closure_accepted = sum(
        1 for candidate in accepted_paths
        if (
            candidate['kind'] == 'endpoint_to_existing_seam_vertex'
            and candidate.get('same_component_loop_closure')
        )
    )
    junction_gap_candidates = sum(
        1 for candidate in candidates
        if candidate['kind'] == 'junction_gap_closure'
    )
    junction_gap_accepted = sum(
        1 for candidate in accepted_paths
        if candidate['kind'] == 'junction_gap_closure'
    )
    return {
        'enabled': True,
        'max_gap_hops': max_hops,
        'allow_same_component': bool(allow_same_component),
        'min_same_component_loop_size': min_loop,
        'candidates_total': len(candidates),
        'endpoint_to_existing_seam_candidates': endpoint_to_existing_candidates,
        'endpoint_to_existing_seam_accepted': endpoint_to_existing_accepted,
        'endpoint_loop_closure_candidates': endpoint_loop_closure_candidates,
        'endpoint_loop_closure_accepted': endpoint_loop_closure_accepted,
        'endpoint_to_existing_loop_closure_candidates': endpoint_to_existing_loop_closure_candidates,
        'endpoint_to_existing_loop_closure_accepted': endpoint_to_existing_loop_closure_accepted,
        'junction_gap_candidates': junction_gap_candidates,
        'junction_gap_accepted': junction_gap_accepted,
        'accepted_paths_count': len(accepted_paths),
        'accepted_edges_count': len(accepted_edge_keys),
        'rejected_same_component': counters['rejected_same_component'],
        'rejected_endpoint_to_existing_same_component': counters[
            'rejected_endpoint_to_existing_same_component'
        ],
        'rejected_endpoint_same_component_too_short': counters[
            'rejected_endpoint_same_component_too_short'
        ],
        'rejected_endpoint_to_existing_same_component_too_short': counters[
            'rejected_endpoint_to_existing_same_component_too_short'
        ],
        'rejected_same_component_no_existing_path': counters[
            'rejected_same_component_no_existing_path'
        ],
        'rejected_existing_seam_internal': counters['rejected_existing_seam_internal'],
        'rejected_internal_seam_vertex': counters['rejected_internal_seam_vertex'],
        'rejected_conflict_consumed_endpoint': rejected_conflict_consumed_endpoint,
        'rejected_junction_gap_no_high_degree_endpoint': counters[
            'rejected_junction_gap_no_high_degree_endpoint'
        ],
        'rejected_junction_gap_same_component_too_short': counters[
            'rejected_junction_gap_same_component_too_short'
        ],
        'rejected_junction_gap_internal_seam_vertex': counters[
            'rejected_junction_gap_internal_seam_vertex'
        ],
        'rejected_junction_gap_existing_seam_edge': counters[
            'rejected_junction_gap_existing_seam_edge'
        ],
        'rejected_junction_gap_reserved_edge': counters[
            'rejected_junction_gap_reserved_edge'
        ],
        'rejected_no_path': counters['rejected_no_path'],
        'accepted_paths': tuple(accepted_paths),
    }


def _editable_gap_fill_empty_result(enabled, max_gap_hops):
    return {
        'enabled': bool(enabled),
        'max_gap_hops': int(max_gap_hops),
        'allow_same_component': False,
        'min_same_component_loop_size': 8,
        'candidates_total': 0,
        'endpoint_to_existing_seam_candidates': 0,
        'endpoint_to_existing_seam_accepted': 0,
        'endpoint_loop_closure_candidates': 0,
        'endpoint_loop_closure_accepted': 0,
        'endpoint_to_existing_loop_closure_candidates': 0,
        'endpoint_to_existing_loop_closure_accepted': 0,
        'junction_gap_candidates': 0,
        'junction_gap_accepted': 0,
        'accepted_paths_count': 0,
        'accepted_edges_count': 0,
        'rejected_same_component': 0,
        'rejected_endpoint_to_existing_same_component': 0,
        'rejected_endpoint_same_component_too_short': 0,
        'rejected_endpoint_to_existing_same_component_too_short': 0,
        'rejected_same_component_no_existing_path': 0,
        'rejected_existing_seam_internal': 0,
        'rejected_internal_seam_vertex': 0,
        'rejected_conflict_consumed_endpoint': 0,
        'rejected_junction_gap_no_high_degree_endpoint': 0,
        'rejected_junction_gap_same_component_too_short': 0,
        'rejected_junction_gap_internal_seam_vertex': 0,
        'rejected_junction_gap_existing_seam_edge': 0,
        'rejected_junction_gap_reserved_edge': 0,
        'rejected_no_path': 0,
        'accepted_paths': tuple(),
    }


def _bounded_editable_paths(adjacency, source, target, max_hops):
    paths = []
    stack = [(int(source), (int(source),))]
    while stack:
        current, path = stack.pop()
        hops = len(path) - 1
        if hops >= max_hops:
            continue
        for neighbor in sorted(adjacency.get(current, ())):
            neighbor = int(neighbor)
            if neighbor in path:
                continue
            next_path = path + (neighbor,)
            if neighbor == int(target):
                paths.append(next_path)
            else:
                stack.append((neighbor, next_path))
    return paths


def _editable_gap_candidate(mesh, path, kind, start, target, *, same_component_loop_closure=False):
    edge_keys = _path_edge_keys(path)
    consumed_endpoint_vertices = [] if kind == 'junction_gap_closure' else [int(start)]
    if kind == 'endpoint_to_endpoint':
        consumed_endpoint_vertices.append(int(target))
    return {
        'kind': str(kind),
        'start_vertex': int(start),
        'target_vertex': int(target),
        'vertices': [int(vertex) for vertex in path],
        'edges': [[int(edge_key[0]), int(edge_key[1])] for edge_key in edge_keys],
        'hop_count': len(edge_keys),
        'total_length': _editable_path_length(mesh, path),
        'consumed_endpoint_vertices': consumed_endpoint_vertices,
        'same_component_loop_closure': bool(same_component_loop_closure),
    }


def _editable_path_length(mesh, path):
    total = 0.0
    for left, right in zip(path, path[1:]):
        left_position = _vertex_position(mesh, int(left))
        right_position = _vertex_position(mesh, int(right))
        if left_position is None or right_position is None:
            total += 1.0
        else:
            total += _distance(left_position, right_position)
    return float(total)


def apply_editable_dangling_seam_cleanup(
    mesh,
    *,
    enabled=True,
    max_dangling_edges=1,
    protect_boundary_vertices=True,
    allow_remove_entire_component=False,
):
    if isinstance(max_dangling_edges, bool) or int(max_dangling_edges) < 1:
        raise ValueError('max_dangling_edges must be an integer greater than or equal to 1')

    max_edges = int(max_dangling_edges)
    if not enabled:
        return _editable_dangling_cleanup_empty_result(
            False,
            max_edges,
            protect_boundary_vertices,
        )

    edge_items, edge_by_key, _ = _mesh_edge_lookup(mesh)
    seam_degree, seam_adjacency = _seam_topology_from_mesh_edges(edge_items)
    component_id_of = _seam_component_ids(seam_adjacency)
    seam_edge_keys = {
        key for _, key, edge in edge_items
        if bool(edge.use_seam)
    }
    component_edge_counts = {}
    component_edge_keys = {}
    for edge_key in seam_edge_keys:
        component_id = component_id_of.get(edge_key[0])
        if component_id is None:
            continue
        component_edge_counts[component_id] = component_edge_counts.get(component_id, 0) + 1
        component_edge_keys.setdefault(component_id, set()).add(edge_key)
    boundary_vertices = (
        _mesh_boundary_vertices(mesh)
        if protect_boundary_vertices
        else set()
    )

    counters = {
        'rejected_too_long': 0,
        'rejected_boundary_protected': 0,
        'rejected_entire_component': 0,
        'rejected_terminal_not_junction': 0,
        'rejected_conflict_removed_edge': 0,
        'rejected_isolated_path_too_long': 0,
        'rejected_isolated_path_boundary_protected': 0,
        'rejected_isolated_path_not_simple': 0,
    }
    candidates_total = 0
    isolated_path_candidates = 0
    removable_candidates = []

    for start in sorted(vertex for vertex, degree in seam_degree.items() if int(degree) == 1):
        candidates_total += 1
        report = _dangling_branch_candidate(
            start,
            seam_adjacency,
            seam_degree,
            seam_edge_keys,
            edge_by_key,
            max_edges,
            boundary_vertices,
            component_id_of,
            component_edge_counts,
            protect_boundary_vertices=bool(protect_boundary_vertices),
            allow_remove_entire_component=bool(allow_remove_entire_component),
        )
        reason = report.get('rejection_reason')
        if reason is None:
            removable_candidates.append(report)
        elif reason in counters:
            counters[reason] += 1

    for component_id in sorted(component_edge_keys):
        isolated_path_candidates += 1
        report = _isolated_path_component_candidate(
            component_id,
            component_edge_keys[component_id],
            seam_adjacency,
            seam_degree,
            edge_by_key,
            max_edges,
            boundary_vertices,
            protect_boundary_vertices=bool(protect_boundary_vertices),
        )
        reason = report.get('rejection_reason')
        if reason is None:
            removable_candidates.append(report)
        elif reason in counters:
            counters[reason] += 1

    kind_priority = {
        'dangling_branch': 0,
        'isolated_path_component': 1,
    }
    removable_candidates.sort(
        key=lambda item: (
            item['length'],
            kind_priority.get(item.get('kind'), 99),
            item['start_vertex'],
            item['terminal_vertex'],
            tuple(tuple(edge_key) for edge_key in item['edge_keys']),
        )
    )

    removed_edge_keys = set()
    removed_edges_by_component = {}
    removed_branches = []
    for candidate in removable_candidates:
        edge_keys = {tuple(edge_key) for edge_key in candidate['edge_keys']}
        if edge_keys & removed_edge_keys:
            counters['rejected_conflict_removed_edge'] += 1
            continue
        component_id = candidate.get('component_id')
        component_edge_count = int(candidate.get('component_edge_count', 0))
        if (
            candidate.get('kind') != 'isolated_path_component'
            and not allow_remove_entire_component
            and component_id is not None
            and component_edge_count > 0
            and removed_edges_by_component.get(component_id, 0) + len(edge_keys) >= component_edge_count
        ):
            counters['rejected_entire_component'] += 1
            continue
        for edge_key in edge_keys:
            edge_by_key[edge_key][1].use_seam = False
        removed_edge_keys.update(edge_keys)
        if component_id is not None:
            removed_edges_by_component[component_id] = (
                removed_edges_by_component.get(component_id, 0) + len(edge_keys)
            )
        removed_branches.append(candidate)

    return {
        'enabled': True,
        'max_dangling_edges': max_edges,
        'protect_boundary_vertices': bool(protect_boundary_vertices),
        'allow_remove_entire_component': bool(allow_remove_entire_component),
        'candidates_total': candidates_total,
        'isolated_path_candidates': isolated_path_candidates,
        'isolated_paths_removed': sum(
            1 for branch in removed_branches
            if branch.get('kind') == 'isolated_path_component'
        ),
        'isolated_path_edges_removed': sum(
            int(branch.get('length', 0)) for branch in removed_branches
            if branch.get('kind') == 'isolated_path_component'
        ),
        'removed_branches_count': len(removed_branches),
        'removed_edges_count': len(removed_edge_keys),
        'rejected_too_long': counters['rejected_too_long'],
        'rejected_boundary_protected': counters['rejected_boundary_protected'],
        'rejected_entire_component': counters['rejected_entire_component'],
        'rejected_terminal_not_junction': counters['rejected_terminal_not_junction'],
        'rejected_conflict_removed_edge': counters['rejected_conflict_removed_edge'],
        'rejected_isolated_path_too_long': counters['rejected_isolated_path_too_long'],
        'rejected_isolated_path_boundary_protected': counters[
            'rejected_isolated_path_boundary_protected'
        ],
        'rejected_isolated_path_not_simple': counters['rejected_isolated_path_not_simple'],
        'removed_branches': tuple(removed_branches),
    }


def _editable_dangling_cleanup_empty_result(enabled, max_dangling_edges, protect_boundary_vertices):
    return {
        'enabled': bool(enabled),
        'max_dangling_edges': int(max_dangling_edges),
        'protect_boundary_vertices': bool(protect_boundary_vertices),
        'allow_remove_entire_component': False,
        'candidates_total': 0,
        'isolated_path_candidates': 0,
        'isolated_paths_removed': 0,
        'isolated_path_edges_removed': 0,
        'removed_branches_count': 0,
        'removed_edges_count': 0,
        'rejected_too_long': 0,
        'rejected_boundary_protected': 0,
        'rejected_entire_component': 0,
        'rejected_terminal_not_junction': 0,
        'rejected_conflict_removed_edge': 0,
        'rejected_isolated_path_too_long': 0,
        'rejected_isolated_path_boundary_protected': 0,
        'rejected_isolated_path_not_simple': 0,
        'removed_branches': tuple(),
    }


def _dangling_branch_candidate(
    start,
    seam_adjacency,
    seam_degree,
    seam_edge_keys,
    edge_by_key,
    max_edges,
    boundary_vertices,
    component_id_of,
    component_edge_counts,
    *,
    protect_boundary_vertices,
    allow_remove_entire_component,
):
    if protect_boundary_vertices and int(start) in boundary_vertices:
        return {'rejection_reason': 'rejected_boundary_protected'}

    neighbors = sorted(seam_adjacency.get(start, ()))
    if len(neighbors) != 1:
        return {'rejection_reason': 'rejected_terminal_not_junction'}
    component_id = component_id_of.get(int(start))
    component_edge_count = int(component_edge_counts.get(component_id, 0))

    previous = int(start)
    current = int(neighbors[0])
    path_vertices = [int(start), current]
    edge_keys = [_edge_key(previous, current)]

    while True:
        if len(edge_keys) > max_edges:
            return {'rejection_reason': 'rejected_too_long'}

        degree = int(seam_degree.get(current, 0))
        if degree >= 3:
            return _dangling_branch_report(
                start,
                current,
                edge_keys,
                edge_by_key,
                component_id,
                component_edge_count,
            )

        if degree == 1:
            if protect_boundary_vertices and current in boundary_vertices:
                return {'rejection_reason': 'rejected_boundary_protected'}
            if allow_remove_entire_component:
                return _dangling_branch_report(
                    start,
                    current,
                    edge_keys,
                    edge_by_key,
                    component_id,
                    component_edge_count,
                )
            return {'rejection_reason': 'rejected_entire_component'}

        if degree != 2:
            return {'rejection_reason': 'rejected_terminal_not_junction'}

        next_vertices = [
            int(vertex) for vertex in sorted(seam_adjacency.get(current, ()))
            if int(vertex) != previous
        ]
        if len(next_vertices) != 1:
            return {'rejection_reason': 'rejected_terminal_not_junction'}
        next_vertex = next_vertices[0]
        next_edge_key = _edge_key(current, next_vertex)
        if next_edge_key not in seam_edge_keys or next_vertex in path_vertices:
            return {'rejection_reason': 'rejected_terminal_not_junction'}
        edge_keys.append(next_edge_key)
        path_vertices.append(next_vertex)
        previous, current = current, next_vertex


def _dangling_branch_report(start, terminal, edge_keys, edge_by_key, component_id, component_edge_count):
    return {
        'kind': 'dangling_branch',
        'start_vertex': int(start),
        'terminal_vertex': int(terminal),
        'edge_keys': [[int(edge_key[0]), int(edge_key[1])] for edge_key in edge_keys],
        'edge_indices': [int(edge_by_key[edge_key][0]) for edge_key in edge_keys],
        'length': len(edge_keys),
        'component_id': component_id,
        'component_edge_count': int(component_edge_count),
    }


def _isolated_path_component_candidate(
    component_id,
    component_edge_keys,
    seam_adjacency,
    seam_degree,
    edge_by_key,
    max_edges,
    boundary_vertices,
    *,
    protect_boundary_vertices,
):
    edge_keys = sorted(tuple(edge_key) for edge_key in component_edge_keys)
    edge_count = len(edge_keys)
    if edge_count < 1:
        return {'rejection_reason': 'rejected_isolated_path_not_simple'}
    if edge_count > max_edges:
        return {'rejection_reason': 'rejected_isolated_path_too_long'}

    vertices = sorted({
        int(vertex)
        for edge_key in edge_keys
        for vertex in edge_key
    })
    endpoint_vertices = [
        vertex for vertex in vertices
        if int(seam_degree.get(vertex, 0)) == 1
    ]
    if (
        len(endpoint_vertices) != 2
        or any(int(seam_degree.get(vertex, 0)) >= 3 for vertex in vertices)
        or any(
            int(seam_degree.get(vertex, 0)) not in (1, 2)
            for vertex in vertices
        )
    ):
        return {'rejection_reason': 'rejected_isolated_path_not_simple'}

    if protect_boundary_vertices and any(vertex in boundary_vertices for vertex in endpoint_vertices):
        return {'rejection_reason': 'rejected_isolated_path_boundary_protected'}

    return {
        'kind': 'isolated_path_component',
        'start_vertex': int(endpoint_vertices[0]),
        'terminal_vertex': int(endpoint_vertices[1]),
        'edge_keys': [[int(edge_key[0]), int(edge_key[1])] for edge_key in edge_keys],
        'edge_indices': [int(edge_by_key[edge_key][0]) for edge_key in edge_keys],
        'length': edge_count,
        'component_id': component_id,
        'component_edge_count': edge_count,
    }


def _mesh_boundary_vertices(mesh):
    polygons = getattr(mesh, 'polygons', None)
    if not polygons:
        return set()

    edge_face_counts = {}
    for polygon in polygons:
        vertices = list(getattr(polygon, 'vertices', ()))
        if len(vertices) < 2:
            continue
        for index, vertex in enumerate(vertices):
            next_vertex = vertices[(index + 1) % len(vertices)]
            key = _edge_key(vertex, next_vertex)
            edge_face_counts[key] = edge_face_counts.get(key, 0) + 1

    boundary_vertices = set()
    for edge_key, count in edge_face_counts.items():
        if count == 1:
            boundary_vertices.update(edge_key)
    return boundary_vertices


def apply_editable_seam_mirror(
    mesh,
    *,
    direction,
    axis='X',
    tolerance=1e-4,
    enabled=True,
    skip_center_edges=True,
):
    if direction not in ('NEGATIVE_TO_POSITIVE', 'POSITIVE_TO_NEGATIVE'):
        raise ValueError(
            "direction must be 'NEGATIVE_TO_POSITIVE' or 'POSITIVE_TO_NEGATIVE'"
        )
    if isinstance(tolerance, bool) or float(tolerance) <= 0.0:
        raise ValueError('tolerance must be greater than 0')

    tolerance = float(tolerance)
    axis_index = _axis_index(axis)
    normalized_axis = str(axis).upper()
    if not enabled:
        return _editable_seam_mirror_empty_result(
            False,
            direction,
            normalized_axis,
            tolerance,
        )

    edge_items, edge_by_key, _ = _mesh_edge_lookup(mesh)
    original_seam_keys = {
        key for _, key, edge in edge_items
        if bool(edge.use_seam)
    }
    mirror_vertex_of = _mirrored_vertex_lookup(mesh, axis_index, tolerance)

    source_seam_edges = 0
    mirrored_edges_added = 0
    mirrored_edges_already_present = 0
    skipped_center_edges = 0
    unmatched_vertices = 0
    missing_mirrored_edges = 0
    mirrored_edges = []

    for edge_index, edge_key, edge in sorted(edge_items, key=lambda item: (item[1], item[0])):
        if edge_key not in original_seam_keys:
            continue

        left_position = _vertex_position(mesh, edge_key[0])
        right_position = _vertex_position(mesh, edge_key[1])
        if left_position is None or right_position is None:
            unmatched_vertices += 1
            continue

        midpoint = (left_position[axis_index] + right_position[axis_index]) / 2.0
        if abs(midpoint) <= tolerance:
            if skip_center_edges:
                skipped_center_edges += 1
            continue
        if direction == 'NEGATIVE_TO_POSITIVE' and midpoint >= -tolerance:
            continue
        if direction == 'POSITIVE_TO_NEGATIVE' and midpoint <= tolerance:
            continue

        source_seam_edges += 1
        mirror_left = mirror_vertex_of.get(edge_key[0])
        mirror_right = mirror_vertex_of.get(edge_key[1])
        if mirror_left is None or mirror_right is None:
            unmatched_vertices += 1
            continue

        mirrored_key = _edge_key(mirror_left, mirror_right)
        mirrored_edge = edge_by_key.get(mirrored_key)
        if mirrored_edge is None:
            missing_mirrored_edges += 1
            continue

        mirrored_edge_index, mirrored_edge_object = mirrored_edge
        if bool(mirrored_edge_object.use_seam):
            mirrored_edges_already_present += 1
        else:
            mirrored_edge_object.use_seam = True
            mirrored_edges_added += 1
            mirrored_edges.append({
                'source_edge_key': [int(edge_key[0]), int(edge_key[1])],
                'source_edge_index': int(edge_index),
                'mirrored_edge_key': [int(mirrored_key[0]), int(mirrored_key[1])],
                'mirrored_edge_index': int(mirrored_edge_index),
            })

    return {
        'enabled': True,
        'direction': direction,
        'axis': normalized_axis,
        'tolerance': tolerance,
        'source_seam_edges': source_seam_edges,
        'mirrored_edges_added': mirrored_edges_added,
        'mirrored_edges_already_present': mirrored_edges_already_present,
        'skipped_center_edges': skipped_center_edges,
        'unmatched_vertices': unmatched_vertices,
        'missing_mirrored_edges': missing_mirrored_edges,
        'mirrored_edges': tuple(mirrored_edges),
    }


def _editable_seam_mirror_empty_result(enabled, direction, axis, tolerance):
    return {
        'enabled': bool(enabled),
        'direction': direction,
        'axis': axis,
        'tolerance': float(tolerance),
        'source_seam_edges': 0,
        'mirrored_edges_added': 0,
        'mirrored_edges_already_present': 0,
        'skipped_center_edges': 0,
        'unmatched_vertices': 0,
        'missing_mirrored_edges': 0,
        'mirrored_edges': tuple(),
    }


def _axis_index(axis):
    normalized = str(axis).upper()
    if normalized == 'X':
        return 0
    if normalized == 'Y':
        return 1
    if normalized == 'Z':
        return 2
    raise ValueError("axis must be 'X', 'Y', or 'Z'")


def _mirrored_vertex_lookup(mesh, axis_index, tolerance):
    positions_by_vertex = {}
    buckets = {}
    for vertex_index in range(len(getattr(mesh, 'vertices', ()))):
        position = _vertex_position(mesh, vertex_index)
        if position is None:
            continue
        positions_by_vertex[int(vertex_index)] = position
        buckets.setdefault(_mirror_bucket_key(position, tolerance), []).append(int(vertex_index))

    lookup = {}
    for vertex_index, position in positions_by_vertex.items():
        mirrored_position = list(position)
        mirrored_position[axis_index] = -mirrored_position[axis_index]
        lookup[vertex_index] = _find_mirrored_vertex(
            tuple(mirrored_position),
            positions_by_vertex,
            buckets,
            tolerance,
        )
    return lookup


def _find_mirrored_vertex(target_position, positions_by_vertex, buckets, tolerance):
    base_key = _mirror_bucket_key(target_position, tolerance)
    candidates = []
    for x_offset in (-1, 0, 1):
        for y_offset in (-1, 0, 1):
            for z_offset in (-1, 0, 1):
                bucket_key = (
                    base_key[0] + x_offset,
                    base_key[1] + y_offset,
                    base_key[2] + z_offset,
                )
                candidates.extend(buckets.get(bucket_key, ()))
    valid = [
        vertex_index for vertex_index in candidates
        if _position_within_tolerance(positions_by_vertex[vertex_index], target_position, tolerance)
    ]
    return min(valid) if valid else None


def _mirror_bucket_key(position, tolerance):
    return tuple(int(math.floor(float(value) / tolerance)) for value in position)


def _position_within_tolerance(position, target_position, tolerance):
    return all(
        abs(float(position[index]) - float(target_position[index])) <= tolerance
        for index in range(3)
    )


def _edge_key(left, right):
    left = int(left)
    right = int(right)
    return (min(left, right), max(left, right))


def apply_seam_keys(
    mesh,
    predicted_keys,
    clear_existing=True,
    accepted_bridge_keys=None,
    accepted_bridge_entries=None,
    enable_local_repair=False,
    fill_small_gaps=True,
    fill_gap_max_hops=2,
):
    edge_by_key = {}
    for edge in mesh.edges:
        v0, v1 = edge.vertices
        edge_by_key[(min(v0, v1), max(v0, v1))] = edge
    originally_marked = {key for key, edge in edge_by_key.items() if bool(edge.use_seam)}

    if clear_existing:
        for edge in mesh.edges:
            edge.use_seam = False

    requested = len(predicted_keys)
    seen = set()
    applied = 0
    ignored_non_original = 0
    duplicates_skipped = 0
    accepted_bridge_set = {
        (min(key[0], key[1]), max(key[0], key[1]))
        for key in (accepted_bridge_keys or [])
    }
    accepted_entries = []
    if accepted_bridge_entries is not None:
        for entry in accepted_bridge_entries:
            if not isinstance(entry, dict):
                continue
            vertex_ids = entry.get('vertex_ids_0based')
            if not _is_vertex_pair(vertex_ids):
                continue
            v0, v1 = vertex_ids
            normalized = (min(v0, v1), max(v0, v1))
            accepted_bridge_set.add(normalized)
            accepted_entries.append((normalized, dict(entry)))
    else:
        for key in sorted(accepted_bridge_set):
            accepted_entries.append((key, {
                'canonical_edge_index': None,
                'vertex_ids_0based': [int(key[0]), int(key[1])],
                'bridge_path_id': None,
                'path_edge_count': None,
                'same_component': False,
                'present_in_final_json': True,
            }))
    accepted_bridge_applied = 0
    accepted_bridge_ignored = 0
    applied_keys = set()
    duplicate_keys = set()

    for key in predicted_keys:
        normalized_key = (min(key[0], key[1]), max(key[0], key[1]))
        if normalized_key in seen:
            duplicates_skipped += 1
            duplicate_keys.add(normalized_key)
            continue
        seen.add(normalized_key)

        edge = edge_by_key.get(normalized_key)
        if edge is None:
            ignored_non_original += 1
            continue

        edge.use_seam = True
        applied += 1
        applied_keys.add(normalized_key)
        if normalized_key in accepted_bridge_set:
            accepted_bridge_applied += 1

    accepted_bridge_trace = []
    for key, entry in accepted_entries:
        edge = edge_by_key.get(key)
        present_in_final_json = bool(entry.get('present_in_final_json', key in seen))
        blender_edge_key_exists = edge is not None
        applied_to_blender = key in applied_keys
        duplicate_or_already_marked = key in duplicate_keys or (key in originally_marked and not clear_existing)
        ignored_reason = None
        if not present_in_final_json:
            ignored_reason = 'not_present_in_final_json'
        elif not blender_edge_key_exists:
            ignored_reason = 'non_original'
        elif duplicate_or_already_marked and not applied_to_blender:
            ignored_reason = 'duplicate_or_already_marked'
        elif not applied_to_blender:
            ignored_reason = 'not_applied'
        if key in accepted_bridge_set and present_in_final_json and not blender_edge_key_exists:
            accepted_bridge_ignored += 1
        accepted_bridge_trace.append({
            'canonical_edge_index': entry.get('canonical_edge_index'),
            'vertex_ids_0based': [int(key[0]), int(key[1])],
            'bridge_path_id': entry.get('bridge_path_id'),
            'path_edge_count': entry.get('path_edge_count'),
            'same_component': bool(entry.get('same_component', False)),
            'present_in_final_json': present_in_final_json,
            'blender_edge_key_exists': blender_edge_key_exists,
            'applied_to_blender': applied_to_blender,
            'ignored_reason': ignored_reason,
            'duplicate_or_already_marked': duplicate_or_already_marked,
        })

    editable_gap_fill_result = apply_editable_shortest_path_gap_fill(
        mesh,
        enabled=bool(enable_local_repair and fill_small_gaps),
        max_gap_hops=fill_gap_max_hops,
    )
    mesh.update()

    return SeamApplyResult(
        requested=requested,
        unique=len(seen),
        applied=applied,
        ignored_non_original=ignored_non_original,
        duplicates_skipped=duplicates_skipped,
        accepted_bridge_edges_present_in_json=len(accepted_bridge_set),
        accepted_bridge_edges_applied=accepted_bridge_applied,
        accepted_bridge_edges_ignored_non_original=accepted_bridge_ignored,
        accepted_bridge_apply_trace=tuple(accepted_bridge_trace),
        editable_gap_fill_result=editable_gap_fill_result,
    )


def _mesh_edge_lookup(mesh):
    edge_items = []
    edge_by_key = {}
    adjacency = {}
    for fallback_index, edge in enumerate(mesh.edges):
        v0, v1 = edge.vertices
        key = (min(v0, v1), max(v0, v1))
        edge_index = int(getattr(edge, 'index', fallback_index))
        edge_items.append((edge_index, key, edge))
        edge_by_key[key] = (edge_index, edge)
        adjacency.setdefault(key[0], set()).add(key[1])
        adjacency.setdefault(key[1], set()).add(key[0])
    return edge_items, edge_by_key, adjacency


def _vertex_position(mesh, vertex_index):
    vertices = getattr(mesh, 'vertices', None)
    if vertices is None or vertex_index < 0 or vertex_index >= len(vertices):
        return None
    co = getattr(vertices[vertex_index], 'co', None)
    if co is None:
        return None
    try:
        return (float(co[0]), float(co[1]), float(co[2]))
    except (TypeError, ValueError, IndexError):
        try:
            return (float(co.x), float(co.y), float(co.z))
        except AttributeError:
            return None


def _vector_sub(left, right):
    return (
        float(left[0]) - float(right[0]),
        float(left[1]) - float(right[1]),
        float(left[2]) - float(right[2]),
    )


def _distance(left, right):
    return _norm(_vector_sub(left, right))


def _norm(vector):
    return (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5


def _path_edge_keys(path):
    return [
        (min(path[index], path[index + 1]), max(path[index], path[index + 1]))
        for index in range(len(path) - 1)
    ]


def _seam_topology_from_mesh_edges(edge_items):
    seam_degree = {}
    seam_adjacency = {}
    for _, key, edge in edge_items:
        if not edge.use_seam:
            continue
        u, v = key
        seam_degree[u] = seam_degree.get(u, 0) + 1
        seam_degree[v] = seam_degree.get(v, 0) + 1
        seam_adjacency.setdefault(u, set()).add(v)
        seam_adjacency.setdefault(v, set()).add(u)
    return seam_degree, seam_adjacency


def _seam_component_ids(seam_adjacency):
    component_id_of = {}
    component_id = 0
    for start in sorted(seam_adjacency):
        if start in component_id_of:
            continue
        stack = [start]
        component_id_of[start] = component_id
        while stack:
            current = stack.pop()
            for neighbor in sorted(seam_adjacency.get(current, ())):
                if neighbor in component_id_of:
                    continue
                component_id_of[neighbor] = component_id
                stack.append(neighbor)
        component_id += 1
    return component_id_of


def _shortest_seam_path_length(seam_adjacency, source, target):
    if source == target:
        return 0
    visited = {source}
    queue = [(source, 0)]
    while queue:
        current, distance = queue.pop(0)
        for neighbor in sorted(seam_adjacency.get(current, ())):
            if neighbor in visited:
                continue
            if neighbor == target:
                return distance + 1
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def format_apply_summary(result):
    parts = [
        f'Marked {result.applied} seam edges. '
        f'Ignored {result.ignored_non_original} triangulation-only edges. '
        f'Skipped {result.duplicates_skipped} duplicates.'
    ]
    if result.accepted_bridge_edges_present_in_json > 0:
        parts.append(
            f' Bridge: {result.accepted_bridge_edges_present_in_json} accepted in JSON, '
            f'{result.accepted_bridge_edges_applied} applied, '
            f'{result.accepted_bridge_edges_ignored_non_original} ignored as non-original.'
        )
    gap = result.editable_gap_fill_result
    if gap and int(gap.get('accepted_paths_count', 0)) > 0:
        paths = int(gap['accepted_paths_count'])
        edges = int(gap['accepted_edges_count'])
        parts.append(f' Gap fill: {paths} paths filled, {edges} edges added.')
    trace = '; '.join(
        (
            f"#{entry['canonical_edge_index']} "
            f"{entry['vertex_ids_0based']} "
            f"{'applied' if entry['applied_to_blender'] else 'ignored:' + str(entry['ignored_reason'])}"
        )
        for entry in result.accepted_bridge_apply_trace
    )
    if trace:
        parts.append(f' Bridge trace: {trace}.')
    return ''.join(parts)


def _is_vertex_pair(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and type(value[0]) is int
        and type(value[1]) is int
        and value[0] >= 0
        and value[1] >= 0
    )
