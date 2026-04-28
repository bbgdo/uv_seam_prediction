import json
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
    blender_local_repair_enabled: bool = False
    blender_local_repair_candidates_total: int = 0
    blender_local_repair_edges_marked: int = 0
    blender_local_repair_edges_rejected: int = 0
    blender_local_repair_candidate_reports: tuple = ()
    human_case_2557_2558_found: bool = False
    human_case_2557_2558_accepted: bool = False
    human_case_2557_2558_marked_seam: bool = False
    human_case_2557_2558_rejection_reason: str | None = None


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


def apply_seam_keys(
    mesh,
    predicted_keys,
    clear_existing=True,
    accepted_bridge_keys=None,
    accepted_bridge_entries=None,
    enable_local_repair=False,
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

    repair = apply_missing_edge_continuity_repair(
        mesh,
        enabled=bool(enable_local_repair),
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
        blender_local_repair_enabled=bool(repair['enabled']),
        blender_local_repair_candidates_total=int(repair['candidates_total']),
        blender_local_repair_edges_marked=int(repair['edges_marked']),
        blender_local_repair_edges_rejected=int(repair['edges_rejected']),
        blender_local_repair_candidate_reports=tuple(repair['candidate_reports']),
        human_case_2557_2558_found=bool(repair['human_case_2557_2558_found']),
        human_case_2557_2558_accepted=bool(repair['human_case_2557_2558_accepted']),
        human_case_2557_2558_marked_seam=bool(repair['human_case_2557_2558_marked_seam']),
        human_case_2557_2558_rejection_reason=repair['human_case_2557_2558_rejection_reason'],
    )


def apply_missing_edge_continuity_repair(mesh, enabled=True, human_case=(2557, 2558)):
    human_key = (min(human_case[0], human_case[1]), max(human_case[0], human_case[1]))
    edge_items = []
    edge_by_key = {}
    for fallback_index, edge in enumerate(mesh.edges):
        v0, v1 = edge.vertices
        key = (min(v0, v1), max(v0, v1))
        edge_index = getattr(edge, 'index', fallback_index)
        edge_items.append((int(edge_index), key, edge))
        edge_by_key[key] = edge

    if not enabled:
        return {
            'enabled': False,
            'candidates_total': 0,
            'edges_marked': 0,
            'edges_rejected': 0,
            'candidate_reports': tuple(),
            'human_case_2557_2558_found': human_key in edge_by_key,
            'human_case_2557_2558_accepted': False,
            'human_case_2557_2558_marked_seam': bool(
                human_key in edge_by_key and edge_by_key[human_key].use_seam
            ),
            'human_case_2557_2558_rejection_reason': 'repair_disabled',
        }

    seam_degree, seam_adjacency = _seam_topology_from_mesh_edges(edge_items)
    component_id_of = _seam_component_ids(seam_adjacency)
    candidate_reports = []
    edges_marked = 0
    human_report = None

    for edge_index, key, edge in edge_items:
        if edge.use_seam:
            if key == human_key:
                human_report = _repair_report(
                    key=key,
                    edge_index=edge_index,
                    seam_degree=seam_degree,
                    component_id_of=component_id_of,
                    seam_adjacency=seam_adjacency,
                    accepted=False,
                    rejection_reason='edge_already_seam',
                    marked_seam=True,
                    human_case_match=True,
                )
            continue

        accepted = False
        rejection_reason = None
        degree_u = seam_degree.get(key[0], 0)
        degree_v = seam_degree.get(key[1], 0)
        if degree_u == 0 or degree_v == 0:
            rejection_reason = 'endpoint_not_seam_vertex'
        elif degree_u != 2 or degree_v != 2:
            rejection_reason = 'degree_pattern_not_phase_2a'
        else:
            accepted = True
            edge.use_seam = True
            edges_marked += 1

        report = _repair_report(
            key=key,
            edge_index=edge_index,
            seam_degree=seam_degree,
            component_id_of=component_id_of,
            seam_adjacency=seam_adjacency,
            accepted=accepted,
            rejection_reason=rejection_reason,
            marked_seam=bool(edge.use_seam),
            human_case_match=(key == human_key),
        )
        candidate_reports.append(report)
        if key == human_key:
            human_report = report

    if human_report is None:
        human_report = {
            'accepted': False,
            'marked_seam': False,
            'rejection_reason': 'edge_not_found',
        }

    return {
        'enabled': True,
        'candidates_total': len(candidate_reports),
        'edges_marked': edges_marked,
        'edges_rejected': len(candidate_reports) - edges_marked,
        'candidate_reports': tuple(candidate_reports),
        'human_case_2557_2558_found': human_key in edge_by_key,
        'human_case_2557_2558_accepted': bool(human_report.get('accepted', False)),
        'human_case_2557_2558_marked_seam': bool(human_report.get('marked_seam', False)),
        'human_case_2557_2558_rejection_reason': human_report.get('rejection_reason'),
    }


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


def _repair_report(
    *,
    key,
    edge_index,
    seam_degree,
    component_id_of,
    seam_adjacency,
    accepted,
    rejection_reason,
    marked_seam,
    human_case_match,
):
    u, v = key
    component_u = component_id_of.get(u)
    component_v = component_id_of.get(v)
    same_component = component_u is not None and component_u == component_v
    loop_size = None
    if same_component:
        shortest = _shortest_seam_path_length(seam_adjacency, u, v)
        if shortest is not None:
            loop_size = shortest + 1
    return {
        'vertex_ids_0based': [int(u), int(v)],
        'blender_edge_index': int(edge_index),
        'seam_degree_u_before': int(seam_degree.get(u, 0)),
        'seam_degree_v_before': int(seam_degree.get(v, 0)),
        'same_component_before': bool(same_component),
        'would_create_loop': bool(same_component),
        'estimated_loop_size_if_available': loop_size,
        'accepted': bool(accepted),
        'rejection_reason': rejection_reason,
        'human_case_match': bool(human_case_match),
        'marked_seam': bool(marked_seam),
    }


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


def write_bridge_apply_debug(json_path, result):
    debug_path = json_path.rsplit('.', 1)[0] + '_bridge_apply_debug.json'
    payload = {
        'accepted_bridge_edges_present_in_json': result.accepted_bridge_edges_present_in_json,
        'accepted_bridge_edges_applied': result.accepted_bridge_edges_applied,
        'accepted_bridge_edges_ignored_non_original': result.accepted_bridge_edges_ignored_non_original,
        'accepted_bridge_apply_trace': list(result.accepted_bridge_apply_trace),
        'blender_local_repair_enabled': result.blender_local_repair_enabled,
        'blender_local_repair_candidates_total': result.blender_local_repair_candidates_total,
        'blender_local_repair_edges_marked': result.blender_local_repair_edges_marked,
        'blender_local_repair_edges_rejected': result.blender_local_repair_edges_rejected,
        'blender_local_repair_candidate_reports': list(result.blender_local_repair_candidate_reports),
        'human_case_2557_2558_found': result.human_case_2557_2558_found,
        'human_case_2557_2558_accepted': result.human_case_2557_2558_accepted,
        'human_case_2557_2558_marked_seam': result.human_case_2557_2558_marked_seam,
        'human_case_2557_2558_rejection_reason': result.human_case_2557_2558_rejection_reason,
    }
    with open(debug_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    return debug_path


def format_apply_summary(result):
    trace = '; '.join(
        (
            f"#{entry['canonical_edge_index']} "
            f"{entry['vertex_ids_0based']} "
            f"{'applied' if entry['applied_to_blender'] else 'ignored:' + str(entry['ignored_reason'])}"
        )
        for entry in result.accepted_bridge_apply_trace
    )
    trace_suffix = f' Bridge trace: {trace}.' if trace else ''
    if result.human_case_2557_2558_marked_seam:
        human_status = 'marked'
    elif result.human_case_2557_2558_found:
        human_status = f"rejected:{result.human_case_2557_2558_rejection_reason}"
    else:
        human_status = 'not found'
    return (
        f'Marked {result.applied} seam edges. '
        f'Ignored {result.ignored_non_original} triangulation-only edges. '
        f'Skipped {result.duplicates_skipped} duplicates. '
        f'Bridge debug: {result.accepted_bridge_edges_present_in_json} accepted in JSON, '
        f'{result.accepted_bridge_edges_applied} applied, '
        f'{result.accepted_bridge_edges_ignored_non_original} ignored as non-original.'
        f' Local repair: {result.blender_local_repair_edges_marked} marked, '
        f'{result.blender_local_repair_edges_rejected} rejected. '
        f'Human case [2557,2558]: {human_status}.'
        f'{trace_suffix}'
    )


def _is_vertex_pair(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and type(value[0]) is int
        and type(value[1]) is int
        and value[0] >= 0
        and value[1] >= 0
    )
