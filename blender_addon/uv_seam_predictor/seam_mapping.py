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
    )


def write_bridge_apply_debug(json_path, result):
    debug_path = json_path.rsplit('.', 1)[0] + '_bridge_apply_debug.json'
    payload = {
        'accepted_bridge_edges_present_in_json': result.accepted_bridge_edges_present_in_json,
        'accepted_bridge_edges_applied': result.accepted_bridge_edges_applied,
        'accepted_bridge_edges_ignored_non_original': result.accepted_bridge_edges_ignored_non_original,
        'accepted_bridge_apply_trace': list(result.accepted_bridge_apply_trace),
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
    return (
        f'Marked {result.applied} seam edges. '
        f'Ignored {result.ignored_non_original} triangulation-only edges. '
        f'Skipped {result.duplicates_skipped} duplicates. '
        f'Bridge debug: {result.accepted_bridge_edges_present_in_json} accepted in JSON, '
        f'{result.accepted_bridge_edges_applied} applied, '
        f'{result.accepted_bridge_edges_ignored_non_original} ignored as non-original.'
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
