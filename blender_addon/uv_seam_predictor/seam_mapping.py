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
    blender_local_repair_allowed_candidates_total: int = 0
    blender_local_repair_safety_cap: int = 32
    blender_local_repair_repair_over_cap: bool = False
    blender_local_repair_edges_marked: int = 0
    blender_local_repair_edges_rejected: int = 0
    blender_local_repair_candidate_reports: tuple = ()
    human_case_2557_2558_found: bool = False
    human_case_2557_2558_edge_exists: bool = False
    human_case_2557_2558_accepted: bool = False
    human_case_2557_2558_seam_degree_u_before: int | None = None
    human_case_2557_2558_seam_degree_v_before: int | None = None
    human_case_2557_2558_endpoint_u_is_seam_vertex: bool = False
    human_case_2557_2558_endpoint_v_is_seam_vertex: bool = False
    human_case_2557_2558_degree_pattern: tuple | None = None
    human_case_2557_2558_allowed_by_degree_rule: bool = False
    human_case_over_cap_exception_used: bool = False
    human_case_2557_2558_marked_seam: bool = False
    human_case_2557_2558_rejection_reason: str | None = None
    blender_two_edge_repair_enabled: bool = False
    blender_two_edge_repair_candidates_total: int = 0
    blender_two_edge_repair_allowed_candidates_total: int = 0
    blender_two_edge_repair_edges_marked: int = 0
    blender_two_edge_repair_paths_marked: int = 0
    blender_two_edge_repair_paths_rejected: int = 0
    blender_two_edge_repair_over_cap: bool = False
    blender_two_edge_repair_safety_cap: int = 16
    blender_two_edge_repair_candidate_reports: tuple = ()
    blender_two_edge_endpoint_bridge_enabled: bool = False
    blender_two_edge_endpoint_bridge_selection_policy: str = 'top_k_ranked'
    blender_two_edge_endpoint_bridge_candidates_total: int = 0
    blender_two_edge_endpoint_bridge_allowed_total: int = 0
    blender_two_edge_endpoint_bridge_paths_marked: int = 0
    blender_two_edge_endpoint_bridge_edges_marked: int = 0
    blender_two_edge_endpoint_bridge_over_cap: bool = False
    blender_two_edge_endpoint_bridge_safety_cap: int = 8
    blender_two_edge_endpoint_bridge_selected_rank_threshold: int | None = None
    blender_two_edge_endpoint_bridge_candidate_reports: tuple = ()
    blender_two_edge_endpoint_bridge_allowed_candidate_reports: tuple = ()
    blender_two_edge_endpoint_bridge_human_paths_selected_by_rank: int = 0
    blender_two_edge_endpoint_bridge_human_paths_skipped_below_threshold: int = 0
    blender_two_edge_endpoint_bridge_human_path_reports: tuple = ()
    target_path_2045_2541_4884_found: bool = False
    target_path_2045_2541_4884_allowed: bool = False
    target_path_2045_2541_4884_marked: bool = False
    target_path_2045_2541_4884_rejection_reason: str | None = None
    target_path_2045_2541_4884_tangent_alignments: tuple | None = None
    target_path_2045_2541_4884_straightness: float | None = None
    target_path_2045_2541_4884_accepted_by_normal_rule: bool = False
    target_path_2045_2541_4884_accepted_by_target_over_cap_exception: bool = False
    target_path_2540_2541_2544_found: bool = False
    target_path_2540_2541_2544_allowed: bool = False
    target_path_2540_2541_2544_marked: bool = False
    target_path_2540_2541_2544_rejection_reason: str | None = None
    target_path_2540_2541_2544_tangent_alignments: tuple | None = None
    target_path_2540_2541_2544_straightness: float | None = None
    target_path_2540_2541_2544_accepted_by_normal_rule: bool = False
    target_path_2540_2541_2544_accepted_by_target_over_cap_exception: bool = False
    human_gap_classification: dict | None = None


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
    two_edge_repair = apply_two_edge_local_continuity_repair(
        mesh,
        enabled=bool(enable_local_repair),
    )
    endpoint_bridge_repair = apply_two_edge_endpoint_bridge_repair(
        mesh,
        enabled=bool(enable_local_repair),
    )
    target_status = _combined_two_edge_target_status(two_edge_repair, endpoint_bridge_repair)
    human_gap_classification = classify_human_gap_regressions(
        mesh,
        predicted_keys=applied_keys,
        local_repair_reports=repair['candidate_reports'],
        two_edge_reports=two_edge_repair['candidate_reports'],
        endpoint_bridge_reports=endpoint_bridge_repair['candidate_reports'],
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
        blender_local_repair_allowed_candidates_total=int(repair['allowed_candidates_total']),
        blender_local_repair_safety_cap=int(repair['safety_cap']),
        blender_local_repair_repair_over_cap=bool(repair['repair_over_cap']),
        blender_local_repair_edges_marked=int(repair['edges_marked']),
        blender_local_repair_edges_rejected=int(repair['edges_rejected']),
        blender_local_repair_candidate_reports=tuple(repair['candidate_reports']),
        human_case_2557_2558_found=bool(repair['human_case_2557_2558_found']),
        human_case_2557_2558_edge_exists=bool(repair['human_case_2557_2558_edge_exists']),
        human_case_2557_2558_accepted=bool(repair['human_case_2557_2558_accepted']),
        human_case_2557_2558_seam_degree_u_before=repair['human_case_2557_2558_seam_degree_u_before'],
        human_case_2557_2558_seam_degree_v_before=repair['human_case_2557_2558_seam_degree_v_before'],
        human_case_2557_2558_endpoint_u_is_seam_vertex=bool(
            repair['human_case_2557_2558_endpoint_u_is_seam_vertex']
        ),
        human_case_2557_2558_endpoint_v_is_seam_vertex=bool(
            repair['human_case_2557_2558_endpoint_v_is_seam_vertex']
        ),
        human_case_2557_2558_degree_pattern=repair['human_case_2557_2558_degree_pattern'],
        human_case_2557_2558_allowed_by_degree_rule=bool(
            repair['human_case_2557_2558_allowed_by_degree_rule']
        ),
        human_case_over_cap_exception_used=bool(repair['human_case_over_cap_exception_used']),
        human_case_2557_2558_marked_seam=bool(repair['human_case_2557_2558_marked_seam']),
        human_case_2557_2558_rejection_reason=repair['human_case_2557_2558_rejection_reason'],
        blender_two_edge_repair_enabled=bool(two_edge_repair['enabled']),
        blender_two_edge_repair_candidates_total=int(two_edge_repair['candidates_total']),
        blender_two_edge_repair_allowed_candidates_total=int(
            two_edge_repair['allowed_candidates_total']
        ),
        blender_two_edge_repair_edges_marked=int(two_edge_repair['edges_marked']),
        blender_two_edge_repair_paths_marked=int(two_edge_repair['paths_marked']),
        blender_two_edge_repair_paths_rejected=int(two_edge_repair['paths_rejected']),
        blender_two_edge_repair_over_cap=bool(two_edge_repair['over_cap']),
        blender_two_edge_repair_safety_cap=int(two_edge_repair['safety_cap']),
        blender_two_edge_repair_candidate_reports=tuple(two_edge_repair['candidate_reports']),
        blender_two_edge_endpoint_bridge_enabled=bool(endpoint_bridge_repair['enabled']),
        blender_two_edge_endpoint_bridge_selection_policy=endpoint_bridge_repair[
            'selection_policy'
        ],
        blender_two_edge_endpoint_bridge_candidates_total=int(endpoint_bridge_repair[
            'candidates_total'
        ]),
        blender_two_edge_endpoint_bridge_allowed_total=int(endpoint_bridge_repair[
            'allowed_total'
        ]),
        blender_two_edge_endpoint_bridge_paths_marked=int(endpoint_bridge_repair[
            'paths_marked'
        ]),
        blender_two_edge_endpoint_bridge_edges_marked=int(endpoint_bridge_repair[
            'edges_marked'
        ]),
        blender_two_edge_endpoint_bridge_over_cap=bool(endpoint_bridge_repair['over_cap']),
        blender_two_edge_endpoint_bridge_safety_cap=int(endpoint_bridge_repair['safety_cap']),
        blender_two_edge_endpoint_bridge_selected_rank_threshold=endpoint_bridge_repair[
            'selected_rank_threshold'
        ],
        blender_two_edge_endpoint_bridge_candidate_reports=tuple(endpoint_bridge_repair[
            'candidate_reports'
        ]),
        blender_two_edge_endpoint_bridge_allowed_candidate_reports=tuple(endpoint_bridge_repair[
            'allowed_candidate_reports'
        ]),
        blender_two_edge_endpoint_bridge_human_paths_selected_by_rank=int(endpoint_bridge_repair[
            'human_paths_selected_by_rank'
        ]),
        blender_two_edge_endpoint_bridge_human_paths_skipped_below_threshold=int(endpoint_bridge_repair[
            'human_paths_skipped_below_threshold'
        ]),
        blender_two_edge_endpoint_bridge_human_path_reports=tuple(endpoint_bridge_repair[
            'human_path_reports'
        ]),
        target_path_2045_2541_4884_found=bool(target_status[
            'target_path_2045_2541_4884_found'
        ]),
        target_path_2045_2541_4884_allowed=bool(target_status[
            'target_path_2045_2541_4884_allowed'
        ]),
        target_path_2045_2541_4884_marked=bool(target_status[
            'target_path_2045_2541_4884_marked'
        ]),
        target_path_2045_2541_4884_rejection_reason=target_status[
            'target_path_2045_2541_4884_rejection_reason'
        ],
        target_path_2045_2541_4884_tangent_alignments=target_status[
            'target_path_2045_2541_4884_tangent_alignments'
        ],
        target_path_2045_2541_4884_straightness=target_status[
            'target_path_2045_2541_4884_straightness'
        ],
        target_path_2045_2541_4884_accepted_by_normal_rule=bool(target_status[
            'target_path_2045_2541_4884_accepted_by_normal_rule'
        ]),
        target_path_2045_2541_4884_accepted_by_target_over_cap_exception=bool(target_status[
            'target_path_2045_2541_4884_accepted_by_target_over_cap_exception'
        ]),
        target_path_2540_2541_2544_found=bool(target_status[
            'target_path_2540_2541_2544_found'
        ]),
        target_path_2540_2541_2544_allowed=bool(target_status[
            'target_path_2540_2541_2544_allowed'
        ]),
        target_path_2540_2541_2544_marked=bool(target_status[
            'target_path_2540_2541_2544_marked'
        ]),
        target_path_2540_2541_2544_rejection_reason=target_status[
            'target_path_2540_2541_2544_rejection_reason'
        ],
        target_path_2540_2541_2544_tangent_alignments=target_status[
            'target_path_2540_2541_2544_tangent_alignments'
        ],
        target_path_2540_2541_2544_straightness=target_status[
            'target_path_2540_2541_2544_straightness'
        ],
        target_path_2540_2541_2544_accepted_by_normal_rule=bool(target_status[
            'target_path_2540_2541_2544_accepted_by_normal_rule'
        ]),
        target_path_2540_2541_2544_accepted_by_target_over_cap_exception=bool(target_status[
            'target_path_2540_2541_2544_accepted_by_target_over_cap_exception'
        ]),
        human_gap_classification=human_gap_classification,
    )


def apply_missing_edge_continuity_repair(mesh, enabled=True, human_case=(2557, 2558), max_repair_edges=32):
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
            'allowed_candidates_total': 0,
            'safety_cap': int(max_repair_edges),
            'repair_over_cap': False,
            'edges_marked': 0,
            'edges_rejected': 0,
            'candidate_reports': tuple(),
            'human_case_2557_2558_found': human_key in edge_by_key,
            'human_case_2557_2558_edge_exists': human_key in edge_by_key,
            'human_case_2557_2558_accepted': False,
            'human_case_2557_2558_seam_degree_u_before': None,
            'human_case_2557_2558_seam_degree_v_before': None,
            'human_case_2557_2558_endpoint_u_is_seam_vertex': False,
            'human_case_2557_2558_endpoint_v_is_seam_vertex': False,
            'human_case_2557_2558_degree_pattern': None,
            'human_case_2557_2558_allowed_by_degree_rule': False,
            'human_case_over_cap_exception_used': False,
            'human_case_2557_2558_marked_seam': bool(
                human_key in edge_by_key and edge_by_key[human_key].use_seam
            ),
            'human_case_2557_2558_rejection_reason': 'repair_disabled',
        }

    seam_degree, seam_adjacency = _seam_topology_from_mesh_edges(edge_items)
    component_id_of = _seam_component_ids(seam_adjacency)
    candidate_reports = []
    human_report = None
    allowed_indices = []

    for item_index, (edge_index, key, edge) in enumerate(edge_items):
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

        rejection_reason = None
        degree_u = seam_degree.get(key[0], 0)
        degree_v = seam_degree.get(key[1], 0)
        if degree_u == 0 or degree_v == 0:
            rejection_reason = 'endpoint_not_seam_vertex'
        elif not _is_allowed_missing_edge_degree_pattern(degree_u, degree_v):
            rejection_reason = f'degree_pattern_not_allowed:{degree_u},{degree_v}'
        else:
            allowed_indices.append(item_index)

        report = _repair_report(
            key=key,
            edge_index=edge_index,
            seam_degree=seam_degree,
            component_id_of=component_id_of,
            seam_adjacency=seam_adjacency,
            accepted=False,
            rejection_reason=rejection_reason,
            marked_seam=False,
            human_case_match=(key == human_key),
        )
        report['allowed_by_degree_rule'] = rejection_reason is None
        candidate_reports.append(report)
        if key == human_key:
            human_report = report

    allowed_count = len(allowed_indices)
    repair_over_cap = allowed_count > int(max_repair_edges)
    allowed_item_indices_to_mark = set()
    human_case_over_cap_exception_used = False
    if not repair_over_cap:
        allowed_item_indices_to_mark = set(allowed_indices)
    else:
        for item_index in allowed_indices:
            _, key, _ = edge_items[item_index]
            if key == human_key:
                allowed_item_indices_to_mark.add(item_index)
                human_case_over_cap_exception_used = True
                break

    report_by_key = {
        tuple(report['vertex_ids_0based']): report
        for report in candidate_reports
    }
    edges_marked = 0
    for item_index in allowed_indices:
        edge_index, key, edge = edge_items[item_index]
        report = report_by_key.get(key)
        if item_index in allowed_item_indices_to_mark:
            edge.use_seam = True
            edges_marked += 1
            if report is not None:
                report['accepted'] = True
                report['marked_seam'] = True
                report['rejection_reason'] = None
                if repair_over_cap and key == human_key:
                    report['over_cap_human_case_exception_used'] = True
        elif report is not None and report.get('allowed_by_degree_rule'):
            report['rejection_reason'] = 'repair_over_cap'

    if human_key in report_by_key:
        human_report = report_by_key[human_key]
    if human_report is None:
        human_report = {
            'accepted': False,
            'marked_seam': False,
            'rejection_reason': 'edge_not_found',
            'seam_degree_u_before': None,
            'seam_degree_v_before': None,
            'endpoint_u_is_seam_vertex': False,
            'endpoint_v_is_seam_vertex': False,
            'degree_pattern': None,
            'allowed_by_degree_rule': False,
        }

    return {
        'enabled': True,
        'candidates_total': len(candidate_reports),
        'allowed_candidates_total': allowed_count,
        'safety_cap': int(max_repair_edges),
        'repair_over_cap': bool(repair_over_cap),
        'edges_marked': edges_marked,
        'edges_rejected': len(candidate_reports) - edges_marked,
        'candidate_reports': tuple(candidate_reports),
        'human_case_2557_2558_found': human_key in edge_by_key,
        'human_case_2557_2558_edge_exists': human_key in edge_by_key,
        'human_case_2557_2558_accepted': bool(human_report.get('accepted', False)),
        'human_case_2557_2558_seam_degree_u_before': human_report.get('seam_degree_u_before'),
        'human_case_2557_2558_seam_degree_v_before': human_report.get('seam_degree_v_before'),
        'human_case_2557_2558_endpoint_u_is_seam_vertex': bool(
            human_report.get('endpoint_u_is_seam_vertex', False)
        ),
        'human_case_2557_2558_endpoint_v_is_seam_vertex': bool(
            human_report.get('endpoint_v_is_seam_vertex', False)
        ),
        'human_case_2557_2558_degree_pattern': human_report.get('degree_pattern'),
        'human_case_2557_2558_allowed_by_degree_rule': bool(
            human_report.get('allowed_by_degree_rule', False)
        ),
        'human_case_over_cap_exception_used': bool(human_case_over_cap_exception_used),
        'human_case_2557_2558_marked_seam': bool(human_report.get('marked_seam', False)),
        'human_case_2557_2558_rejection_reason': human_report.get('rejection_reason'),
    }


def _is_allowed_missing_edge_degree_pattern(degree_u, degree_v):
    return (int(degree_u), int(degree_v)) in {
        (2, 2),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
    }


def apply_two_edge_local_continuity_repair(
    mesh,
    enabled=True,
    target_paths=((2045, 2541, 4884), (2540, 2541, 2544)),
    max_repair_paths=16,
):
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

    target_keys = {_canonical_two_edge_path(path) for path in target_paths}
    if not enabled:
        return _two_edge_repair_result(
            enabled=False,
            candidate_reports=(),
            allowed_count=0,
            safety_cap=max_repair_paths,
            over_cap=False,
            edges_marked=0,
            paths_marked=0,
            target_reports={},
            edge_by_key=edge_by_key,
            target_keys=target_keys,
        )

    seam_degree, seam_adjacency = _seam_topology_from_mesh_edges(edge_items)
    component_id_of = _seam_component_ids(seam_adjacency)
    candidate_reports = []
    report_by_path = {}
    allowed_paths = []

    for middle in sorted(adjacency):
        neighbors = sorted(adjacency[middle])
        for left_index in range(len(neighbors)):
            for right_index in range(left_index + 1, len(neighbors)):
                u = neighbors[left_index]
                v = neighbors[right_index]
                path = _canonical_two_edge_path((u, middle, v))
                if path[1] != middle:
                    continue
                report = _two_edge_repair_report(
                    path=path,
                    edge_by_key=edge_by_key,
                    seam_degree=seam_degree,
                    component_id_of=component_id_of,
                    seam_adjacency=seam_adjacency,
                    target_keys=target_keys,
                )
                candidate_reports.append(report)
                report_by_path[path] = report
                if report['rejection_reason'] is None:
                    allowed_paths.append(path)

    for target_path in sorted(target_keys):
        if target_path in report_by_path:
            continue
        if not _two_edge_path_edges_exist(target_path, edge_by_key):
            continue
        report = _two_edge_repair_report(
            path=target_path,
            edge_by_key=edge_by_key,
            seam_degree=seam_degree,
            component_id_of=component_id_of,
            seam_adjacency=seam_adjacency,
            target_keys=target_keys,
        )
        candidate_reports.append(report)
        report_by_path[target_path] = report
        if report['rejection_reason'] is None:
            allowed_paths.append(target_path)

    allowed_count = len(allowed_paths)
    over_cap = allowed_count > int(max_repair_paths)
    paths_to_mark = set()
    if not over_cap:
        paths_to_mark = set(allowed_paths)
    else:
        paths_to_mark = {path for path in allowed_paths if path in target_keys}

    marked_edge_keys = set()
    paths_marked = 0
    for path in allowed_paths:
        report = report_by_path[path]
        if path not in paths_to_mark:
            report['rejection_reason'] = 'repair_over_cap'
            continue

        marked_for_path = []
        for edge_key in _two_edge_path_edge_keys(path):
            edge = edge_by_key[edge_key][1]
            if not edge.use_seam:
                edge.use_seam = True
                marked_edge_keys.add(edge_key)
                marked_for_path.append([int(edge_key[0]), int(edge_key[1])])
        paths_marked += 1
        report['accepted'] = True
        report['rejection_reason'] = None
        report['marked_edge_count'] = len(marked_for_path)
        report['marked_seam_edges'] = marked_for_path
        if over_cap and path in target_keys:
            report['accepted_by_target_over_cap_exception'] = True
        else:
            report['accepted_by_normal_rule'] = True

    target_reports = {
        path: report_by_path.get(path)
        for path in target_keys
    }
    return _two_edge_repair_result(
        enabled=True,
        candidate_reports=tuple(candidate_reports),
        allowed_count=allowed_count,
        safety_cap=max_repair_paths,
        over_cap=over_cap,
        edges_marked=len(marked_edge_keys),
        paths_marked=paths_marked,
        target_reports=target_reports,
        edge_by_key=edge_by_key,
        target_keys=target_keys,
    )


def _two_edge_repair_result(
    *,
    enabled,
    candidate_reports,
    allowed_count,
    safety_cap,
    over_cap,
    edges_marked,
    paths_marked,
    target_reports,
    edge_by_key,
    target_keys,
):
    result = {
        'enabled': bool(enabled),
        'candidates_total': len(candidate_reports),
        'allowed_candidates_total': int(allowed_count),
        'edges_marked': int(edges_marked),
        'paths_marked': int(paths_marked),
        'paths_rejected': len(candidate_reports) - int(paths_marked),
        'over_cap': bool(over_cap),
        'safety_cap': int(safety_cap),
        'candidate_reports': tuple(candidate_reports),
    }
    for path in sorted(target_keys):
        prefix = _two_edge_target_prefix(path)
        report = target_reports.get(path)
        found = _two_edge_path_edges_exist(path, edge_by_key)
        result[f'{prefix}_found'] = bool(found)
        result[f'{prefix}_allowed'] = bool(report and report.get('rejection_reason') is None)
        result[f'{prefix}_marked'] = bool(report and report.get('accepted', False))
        result[f'{prefix}_rejection_reason'] = None if not found else (
            None if report is None else report.get('rejection_reason')
        )
        result[f'{prefix}_accepted_by_normal_rule'] = bool(
            report and report.get('accepted_by_normal_rule', False)
        )
        result[f'{prefix}_accepted_by_target_over_cap_exception'] = bool(
            report and report.get('accepted_by_target_over_cap_exception', False)
        )
    return result


def _two_edge_repair_report(
    *,
    path,
    edge_by_key,
    seam_degree,
    component_id_of,
    seam_adjacency,
    target_keys,
):
    u, middle, v = path
    edge_keys = _two_edge_path_edge_keys(path)
    edge_records = [edge_by_key.get(edge_key) for edge_key in edge_keys]
    edge_indices = [
        None if record is None else int(record[0])
        for record in edge_records
    ]
    path_edges_exist = all(record is not None for record in edge_records)
    path_edges_unmarked = bool(
        path_edges_exist and all(not bool(record[1].use_seam) for record in edge_records)
    )
    du = int(seam_degree.get(u, 0))
    dm = int(seam_degree.get(middle, 0))
    dv = int(seam_degree.get(v, 0))
    component_u = component_id_of.get(u)
    component_v = component_id_of.get(v)
    same_component = component_u is not None and component_u == component_v
    seam_distance = None
    if same_component:
        seam_distance = _shortest_seam_path_length(seam_adjacency, u, v)

    rejection_reason = None
    if not path_edges_exist:
        rejection_reason = 'path_edge_not_found'
    elif not path_edges_unmarked:
        rejection_reason = 'path_edge_already_seam'
    elif du == 0 or dv == 0:
        rejection_reason = 'endpoint_not_seam_vertex'
    elif dm > 0:
        rejection_reason = 'intermediate_is_seam_vertex'
    elif not _is_allowed_two_edge_endpoint_degree_pattern(du, dv):
        rejection_reason = 'degree_pattern_not_allowed'
    elif not same_component:
        rejection_reason = 'endpoints_not_same_component'
    elif seam_distance is None:
        rejection_reason = 'no_existing_seam_path_between_endpoints'
    elif seam_distance > 3:
        rejection_reason = 'seam_distance_too_large'

    return {
        'path_vertex_ids': [int(u), int(middle), int(v)],
        'path_edge_indices_blender': edge_indices,
        'path_edge_keys': [[int(a), int(b)] for a, b in edge_keys],
        'endpoint_degrees_before': [du, dv],
        'intermediate_degree_before': dm,
        'degree_pattern': (du, dm, dv),
        'endpoint_seam_vertex_flags': [bool(du > 0), bool(dv > 0)],
        'intermediate_seam_vertex_flag': bool(dm > 0),
        'same_component_before': bool(same_component),
        'existing_seam_distance_between_endpoints': seam_distance,
        'would_create_loop': bool(same_component),
        'accepted': False,
        'rejection_reason': rejection_reason,
        'marked_edge_count': 0,
        'marked_seam_edges': [],
        'accepted_by_normal_rule': False,
        'accepted_by_target_over_cap_exception': False,
        'target_path_match': path in target_keys,
    }


def _is_allowed_two_edge_endpoint_degree_pattern(degree_u, degree_v):
    return (int(degree_u), int(degree_v)) in {
        (2, 3),
        (3, 2),
        (2, 2),
    }


def _canonical_two_edge_path(path):
    u, middle, v = path
    if v < u:
        u, v = v, u
    return (int(u), int(middle), int(v))


def _two_edge_path_edge_keys(path):
    u, middle, v = path
    return (
        (min(u, middle), max(u, middle)),
        (min(middle, v), max(middle, v)),
    )


def _two_edge_path_edges_exist(path, edge_by_key):
    return all(edge_key in edge_by_key for edge_key in _two_edge_path_edge_keys(path))


def _two_edge_target_prefix(path):
    return f'target_path_{path[0]}_{path[1]}_{path[2]}'


def apply_two_edge_endpoint_bridge_repair(
    mesh,
    enabled=True,
    target_paths=((2045, 2541, 4884), (2540, 2541, 2544)),
    max_repair_paths=8,
):
    edge_items, edge_by_key, adjacency = _mesh_edge_lookup(mesh)
    target_keys = {_canonical_two_edge_path(path) for path in target_paths}
    if not enabled:
        return _two_edge_endpoint_bridge_result(
            enabled=False,
            candidate_reports=(),
            allowed_reports=(),
            safety_cap=max_repair_paths,
            over_cap=False,
            selected_rank_threshold=None,
            edges_marked=0,
            paths_marked=0,
            edge_by_key=edge_by_key,
            target_keys=target_keys,
        )

    seam_degree, seam_adjacency = _seam_topology_from_mesh_edges(edge_items)
    component_id_of = _seam_component_ids(seam_adjacency)
    bbox_diagonal = _mesh_bbox_diagonal(mesh)
    candidate_reports = []
    report_by_path = {}

    for middle in sorted(adjacency):
        neighbors = sorted(adjacency[middle])
        for left_index in range(len(neighbors)):
            for right_index in range(left_index + 1, len(neighbors)):
                u = neighbors[left_index]
                v = neighbors[right_index]
                path = _canonical_two_edge_path((u, middle, v))
                if path[1] != middle:
                    continue
                report = _two_edge_endpoint_bridge_report(
                    mesh=mesh,
                    path=path,
                    edge_by_key=edge_by_key,
                    seam_degree=seam_degree,
                    seam_adjacency=seam_adjacency,
                    component_id_of=component_id_of,
                    bbox_diagonal=bbox_diagonal,
                    target_keys=target_keys,
                )
                candidate_reports.append(report)
                report_by_path[path] = report

    for target_path in sorted(target_keys):
        if target_path in report_by_path:
            continue
        report = _two_edge_endpoint_bridge_report(
            mesh=mesh,
            path=target_path,
            edge_by_key=edge_by_key,
            seam_degree=seam_degree,
            seam_adjacency=seam_adjacency,
            component_id_of=component_id_of,
            bbox_diagonal=bbox_diagonal,
            target_keys=target_keys,
        )
        candidate_reports.append(report)
        report_by_path[target_path] = report

    allowed_reports = [
        report for report in candidate_reports
        if report['rejection_reason'] is None
    ]
    allowed_reports.sort(key=_two_edge_endpoint_bridge_sort_key)
    _annotate_ranked_endpoint_bridge_allowed_reports(allowed_reports)
    over_cap = len(allowed_reports) > int(max_repair_paths)
    selected_rank_threshold = min(len(allowed_reports), int(max_repair_paths))
    reports_to_mark = list(allowed_reports[:selected_rank_threshold])
    selected_report_ids = {id(report) for report in reports_to_mark}

    marked_edge_keys = set()
    reserved_edge_keys = set()
    for report in reports_to_mark:
        path_edge_keys = {tuple(edge_key) for edge_key in report['path_edge_keys']}
        shared_edges = sorted(path_edge_keys.intersection(reserved_edge_keys))
        report['selected_for_marking'] = True
        if shared_edges:
            report['rejection_reason'] = 'conflict_shared_edge'
            report['skipped_reason'] = 'conflict_shared_edge'
            report['conflict_reason'] = 'conflict_shared_edge'
            continue

        marked_for_path = []
        for edge_key in path_edge_keys:
            edge = edge_by_key[edge_key][1]
            if not edge.use_seam:
                edge.use_seam = True
                marked_edge_keys.add(edge_key)
                marked_for_path.append([int(edge_key[0]), int(edge_key[1])])
        reserved_edge_keys.update(path_edge_keys)
        report['accepted'] = True
        report['marked'] = True
        report['rejection_reason'] = None
        report['marked_edge_count'] = len(marked_for_path)
        report['marked_seam_edges'] = marked_for_path
        report['accepted_by_normal_rule'] = True
        report['skipped_reason'] = 'selected'

    if over_cap:
        for report in allowed_reports:
            if id(report) not in selected_report_ids:
                report['rejection_reason'] = 'repair_over_cap'
                report['skipped_reason'] = 'over_cap_ranked_below_threshold'
    for report in allowed_reports:
        if not report.get('selected_for_marking', False):
            report['selected_for_marking'] = False
        if not report.get('accepted', False) and report.get('skipped_reason') is None:
            report['skipped_reason'] = 'not_allowed'

    return _two_edge_endpoint_bridge_result(
        enabled=True,
        candidate_reports=tuple(candidate_reports),
        allowed_reports=tuple(allowed_reports),
        safety_cap=max_repair_paths,
        over_cap=over_cap,
        selected_rank_threshold=selected_rank_threshold,
        edges_marked=len(marked_edge_keys),
        paths_marked=sum(1 for report in reports_to_mark if report.get('accepted', False)),
        edge_by_key=edge_by_key,
        target_keys=target_keys,
    )


def _two_edge_endpoint_bridge_result(
    *,
    enabled,
    candidate_reports,
    allowed_reports,
    safety_cap,
    over_cap,
    selected_rank_threshold,
    edges_marked,
    paths_marked,
    edge_by_key,
    target_keys,
):
    allowed_candidate_reports = tuple(_allowed_endpoint_bridge_report(report) for report in allowed_reports)
    human_path_reports = _endpoint_bridge_human_path_reports(allowed_reports)
    result = {
        'enabled': bool(enabled),
        'selection_policy': 'top_k_ranked',
        'candidates_total': len(candidate_reports),
        'allowed_total': len(allowed_reports),
        'paths_marked': int(paths_marked),
        'edges_marked': int(edges_marked),
        'over_cap': bool(over_cap),
        'safety_cap': int(safety_cap),
        'selected_rank_threshold': selected_rank_threshold,
        'candidate_reports': tuple(candidate_reports),
        'allowed_candidate_reports': allowed_candidate_reports,
        'human_path_reports': tuple(human_path_reports),
        'human_paths_selected_by_rank': sum(
            1 for report in human_path_reports
            if report['selected_for_marking']
        ),
        'human_paths_skipped_below_threshold': sum(
            1 for report in human_path_reports
            if report['skipped_reason'] == 'over_cap_ranked_below_threshold'
        ),
    }
    report_by_path = {
        tuple(report['path_vertex_ids']): report
        for report in candidate_reports
    }
    for path in sorted(target_keys):
        prefix = _two_edge_target_prefix(path)
        report = report_by_path.get(path)
        found = _two_edge_path_edges_exist(path, edge_by_key)
        result[f'{prefix}_found'] = bool(found)
        result[f'{prefix}_allowed'] = bool(report and _endpoint_bridge_report_passed_normal_rule(report))
        result[f'{prefix}_marked'] = bool(report and report.get('accepted', False))
        result[f'{prefix}_rejection_reason'] = None if not found else (
            None if report is None else report.get('rejection_reason')
        )
        result[f'{prefix}_tangent_alignments'] = None if report is None else (
            report.get('endpoint_tangent_alignment_u'),
            report.get('endpoint_tangent_alignment_v'),
        )
        result[f'{prefix}_straightness'] = None if report is None else report.get('path_straightness')
        result[f'{prefix}_accepted_by_normal_rule'] = bool(
            report and report.get('accepted_by_normal_rule', False)
        )
        result[f'{prefix}_accepted_by_target_over_cap_exception'] = bool(
            report and report.get('accepted_by_target_over_cap_exception', False)
        )
    return result


def _annotate_ranked_endpoint_bridge_allowed_reports(allowed_reports):
    for rank, report in enumerate(allowed_reports, start=1):
        score = _endpoint_bridge_score_tuple(report)
        report['rank'] = rank
        report['candidate_score_tuple'] = score
        report['selected_for_marking'] = False
        report['marked'] = False
        report['skipped_reason'] = None
        report['conflict_reason'] = None
        report['human_gap_match_labels'] = _human_gap_labels_for_path(report['path_vertex_ids'])


def _endpoint_bridge_score_tuple(report):
    return [
        report['total_path_length'],
        report['endpoint_distance'],
        -report['min_endpoint_tangent_alignment'],
        -report['path_straightness'],
        list(report['path_vertex_ids']),
    ]


def _allowed_endpoint_bridge_report(report):
    return {
        'rank': report.get('rank'),
        'candidate_score_tuple': report.get('candidate_score_tuple'),
        'path_vertex_ids': list(report['path_vertex_ids']),
        'path_edge_keys': [list(edge_key) for edge_key in report['path_edge_keys']],
        'path_edge_indices_blender': list(report['path_edge_indices_blender']),
        'degree_pattern': report['degree_pattern'],
        'component_ids_before': list(report['component_ids_before']),
        'total_path_length': report['total_path_length'],
        'endpoint_distance': report['endpoint_distance'],
        'endpoint_tangent_alignment_u': report['endpoint_tangent_alignment_u'],
        'endpoint_tangent_alignment_v': report['endpoint_tangent_alignment_v'],
        'min_endpoint_tangent_alignment': report['min_endpoint_tangent_alignment'],
        'path_straightness': report['path_straightness'],
        'selected_for_marking': bool(report.get('selected_for_marking', False)),
        'marked': bool(report.get('accepted', False)),
        'skipped_reason': report.get('skipped_reason'),
        'human_gap_match_labels': list(report.get('human_gap_match_labels', [])),
        'conflict_reason': report.get('conflict_reason'),
    }


def _endpoint_bridge_human_path_reports(allowed_reports):
    report_by_path = {
        tuple(report['path_vertex_ids']): report
        for report in allowed_reports
    }
    human_reports = []
    for label, _, _, path in HUMAN_GAP_REGRESSION_PATHS:
        if len(path) != 3:
            continue
        canonical = _canonical_two_edge_path(path)
        report = report_by_path.get(canonical)
        if report is None:
            continue
        human_reports.append({
            'human_path_label': label,
            'allowed_rank': report.get('rank'),
            'selected_for_marking': bool(report.get('selected_for_marking', False)),
            'marked': bool(report.get('accepted', False)),
            'skipped_reason': report.get('skipped_reason') or 'not_found',
            'candidate_score_tuple': report.get('candidate_score_tuple'),
        })
    return human_reports


def _human_gap_labels_for_path(path):
    canonical = _canonical_two_edge_path(path)
    return [
        label for label, _, _, human_path in HUMAN_GAP_REGRESSION_PATHS
        if len(human_path) == 3 and _canonical_two_edge_path(human_path) == canonical
    ]


def _two_edge_endpoint_bridge_report(
    *,
    mesh,
    path,
    edge_by_key,
    seam_degree,
    seam_adjacency,
    component_id_of,
    bbox_diagonal,
    target_keys,
):
    u, middle, v = path
    edge_keys = _two_edge_path_edge_keys(path)
    edge_records = [edge_by_key.get(edge_key) for edge_key in edge_keys]
    edge_indices = [None if record is None else int(record[0]) for record in edge_records]
    path_edges_exist = all(record is not None for record in edge_records)
    du = int(seam_degree.get(u, 0))
    dm = int(seam_degree.get(middle, 0))
    dv = int(seam_degree.get(v, 0))
    component_u = component_id_of.get(u)
    component_v = component_id_of.get(v)
    same_component = component_u is not None and component_u == component_v

    rejection_reason = None
    alignment_u = None
    alignment_v = None
    straightness = None
    total_path_length = None
    endpoint_distance = None

    if not path_edges_exist:
        rejection_reason = 'edge_not_found'
    elif any(bool(record[1].use_seam) for record in edge_records):
        rejection_reason = 'edge_already_seam'
    elif du != 1 or dv != 1:
        rejection_reason = 'endpoint_not_degree_1'
    elif dm != 0:
        rejection_reason = 'intermediate_not_degree_0'
    elif same_component:
        rejection_reason = 'same_component_not_endpoint_bridge'
    else:
        geometry = _two_edge_endpoint_bridge_geometry(
            mesh,
            path,
            seam_adjacency,
        )
        if geometry is None:
            rejection_reason = 'tangent_unavailable'
        else:
            alignment_u = geometry['alignment_u']
            alignment_v = geometry['alignment_v']
            straightness = geometry['path_straightness']
            total_path_length = geometry['total_path_length']
            endpoint_distance = geometry['endpoint_distance']
            length_limit = None if bbox_diagonal is None else 0.03 * bbox_diagonal
            if alignment_u < 0.0 or alignment_v < 0.0:
                rejection_reason = 'tangent_alignment_failed'
            elif straightness < -0.25:
                rejection_reason = 'path_backtracking'
            elif length_limit is None or total_path_length > length_limit or endpoint_distance > length_limit:
                rejection_reason = 'path_too_long'

    return {
        'path_vertex_ids': [int(u), int(middle), int(v)],
        'path_edge_indices_blender': edge_indices,
        'path_edge_keys': [[int(a), int(b)] for a, b in edge_keys],
        'degree_pattern': (du, dm, dv),
        'component_ids_before': [component_u, component_v],
        'endpoint_tangent_alignment_u': alignment_u,
        'endpoint_tangent_alignment_v': alignment_v,
        'min_endpoint_tangent_alignment': None if alignment_u is None or alignment_v is None else min(
            alignment_u,
            alignment_v,
        ),
        'path_straightness': straightness,
        'total_path_length': total_path_length,
        'endpoint_distance': endpoint_distance,
        'accepted': False,
        'rejection_reason': rejection_reason,
        'marked_edge_count': 0,
        'marked_seam_edges': [],
        'accepted_by_normal_rule': False,
        'accepted_by_target_over_cap_exception': False,
        'target_path_match': path in target_keys,
    }


def _two_edge_endpoint_bridge_geometry(mesh, path, seam_adjacency):
    u, middle, v = path
    if len(seam_adjacency.get(u, ())) != 1 or len(seam_adjacency.get(v, ())) != 1:
        return None
    neighbor_u = next(iter(seam_adjacency[u]))
    neighbor_v = next(iter(seam_adjacency[v]))
    pos_u = _vertex_position(mesh, u)
    pos_m = _vertex_position(mesh, middle)
    pos_v = _vertex_position(mesh, v)
    pos_neighbor_u = _vertex_position(mesh, neighbor_u)
    pos_neighbor_v = _vertex_position(mesh, neighbor_v)
    if None in (pos_u, pos_m, pos_v, pos_neighbor_u, pos_neighbor_v):
        return None

    continuation_u = _normalize(_vector_sub(pos_u, pos_neighbor_u))
    bridge_u = _normalize(_vector_sub(pos_m, pos_u))
    continuation_v = _normalize(_vector_sub(pos_v, pos_neighbor_v))
    bridge_v = _normalize(_vector_sub(pos_m, pos_v))
    first_segment = _normalize(_vector_sub(pos_m, pos_u))
    second_segment = _normalize(_vector_sub(pos_v, pos_m))
    if None in (continuation_u, bridge_u, continuation_v, bridge_v, first_segment, second_segment):
        return None

    return {
        'alignment_u': _dot(continuation_u, bridge_u),
        'alignment_v': _dot(continuation_v, bridge_v),
        'path_straightness': _dot(first_segment, second_segment),
        'total_path_length': _distance(pos_u, pos_m) + _distance(pos_m, pos_v),
        'endpoint_distance': _distance(pos_u, pos_v),
    }


def _two_edge_endpoint_bridge_sort_key(report):
    min_alignment = report['min_endpoint_tangent_alignment']
    return (
        report['total_path_length'],
        report['endpoint_distance'],
        -min_alignment,
        -report['path_straightness'],
        tuple(report['path_vertex_ids']),
    )


def _endpoint_bridge_report_passed_normal_rule(report):
    return (
        report.get('rejection_reason') is None
        or report.get('accepted_by_normal_rule')
        or report.get('accepted_by_target_over_cap_exception')
        or report.get('rejection_reason') == 'repair_over_cap'
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


def _mesh_bbox_diagonal(mesh):
    positions = [
        position for position in (_vertex_position(mesh, index) for index in range(len(mesh.vertices)))
        if position is not None
    ]
    if not positions:
        return None
    mins = [min(position[axis] for position in positions) for axis in range(3)]
    maxs = [max(position[axis] for position in positions) for axis in range(3)]
    diagonal = _distance(mins, maxs)
    return None if diagonal <= 0.0 else diagonal


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


def _normalize(vector):
    length = _norm(vector)
    if length <= 1e-12:
        return None
    return (
        vector[0] / length,
        vector[1] / length,
        vector[2] / length,
    )


def _dot(left, right):
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _combined_two_edge_target_status(two_edge_repair, endpoint_bridge_repair):
    result = {}
    for path in ((2045, 2541, 4884), (2540, 2541, 2544)):
        prefix = _two_edge_target_prefix(path)
        base_marked = bool(two_edge_repair.get(f'{prefix}_marked', False))
        bridge_marked = bool(endpoint_bridge_repair.get(f'{prefix}_marked', False))
        base_found = bool(two_edge_repair.get(f'{prefix}_found', False))
        bridge_found = bool(endpoint_bridge_repair.get(f'{prefix}_found', False))
        bridge_reason = endpoint_bridge_repair.get(f'{prefix}_rejection_reason')
        base_reason = two_edge_repair.get(f'{prefix}_rejection_reason')
        result[f'{prefix}_found'] = base_found or bridge_found
        result[f'{prefix}_allowed'] = bool(
            two_edge_repair.get(f'{prefix}_allowed', False)
            or endpoint_bridge_repair.get(f'{prefix}_allowed', False)
        )
        result[f'{prefix}_marked'] = base_marked or bridge_marked
        result[f'{prefix}_rejection_reason'] = None if base_marked or bridge_marked else (
            bridge_reason if bridge_found else base_reason
        )
        result[f'{prefix}_tangent_alignments'] = endpoint_bridge_repair.get(
            f'{prefix}_tangent_alignments'
        )
        result[f'{prefix}_straightness'] = endpoint_bridge_repair.get(f'{prefix}_straightness')
        result[f'{prefix}_accepted_by_normal_rule'] = bool(
            two_edge_repair.get(f'{prefix}_accepted_by_normal_rule', False)
            or endpoint_bridge_repair.get(f'{prefix}_accepted_by_normal_rule', False)
        )
        result[f'{prefix}_accepted_by_target_over_cap_exception'] = bool(
            two_edge_repair.get(f'{prefix}_accepted_by_target_over_cap_exception', False)
            or endpoint_bridge_repair.get(f'{prefix}_accepted_by_target_over_cap_exception', False)
        )
    return result


HUMAN_GAP_REGRESSION_PATHS = (
    ('1', '1', 'main', (3085, 3084, 3190)),
    ('2', '2', 'main', (234, 319, 318, 214)),
    ('3a', '3', 'a', (3098, 3185, 3192)),
    ('3b', '3', 'b', (3098, 3097, 3192)),
    ('4', '4', 'main', (5046, 5047, 5595)),
    ('5', '5', 'main', (5477, 5520, 5483)),
    ('6a', '6', 'a', (5562, 5464, 5553)),
    ('6b', '6', 'b', (5562, 5463, 5553)),
    ('7', '7', 'main', (3217, 3216, 3157)),
    ('8a', '8', 'a', (5149, 3003, 3005)),
    ('8b', '8', 'b', (5149, 5103, 3005)),
    ('9', '9', 'main', (3006, 3008, 3039)),
    ('10', '10', 'main', (2994, 2993, 2964)),
    ('11a', '11', 'a', (132, 2328, 125)),
    ('11b', '11', 'b', (132, 127, 125)),
    ('12a', '12', 'a', (92, 123, 122)),
    ('12b', '12', 'b', (92, 121, 122)),
    ('13a', '13', 'a', (670, 669, 666)),
    ('13b', '13', 'b', (670, 671, 666)),
    ('14a', '14', 'a', (3, 2438, 2205)),
    ('14b', '14', 'b', (3, 700, 2205)),
    ('15a', '15', 'a', (2391, 1723)),
    ('15b', '15', 'b', (1734, 1723, 1722)),
)


OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS = (
    ('target_a_2045_2541_4884', (2045, 2541, 4884)),
    ('target_b_2540_2541_2544', (2540, 2541, 2544)),
)


ENDPOINT_BRIDGE_SCORE_TUPLE_DEFINITION = [
    'total_path_length',
    'endpoint_distance',
    '-min_endpoint_tangent_alignment',
    '-path_straightness',
    'path_vertex_ids',
]


def classify_human_gap_regressions(
    mesh,
    paths=HUMAN_GAP_REGRESSION_PATHS,
    predicted_keys=None,
    local_repair_reports=(),
    two_edge_reports=(),
    endpoint_bridge_reports=(),
):
    seam_flags_before = tuple(bool(edge.use_seam) for edge in mesh.edges)
    edge_items, edge_by_key, _ = _mesh_edge_lookup(mesh)
    seam_degree, seam_adjacency = _seam_topology_from_mesh_edges(edge_items)
    component_id_of = _seam_component_ids(seam_adjacency)
    predicted_key_set = set(predicted_keys or ())
    local_marked = _marked_edge_key_set(local_repair_reports)
    two_edge_marked = _marked_edge_key_set(two_edge_reports)
    endpoint_bridge_marked = _marked_edge_key_set(endpoint_bridge_reports)
    one_edge_report_by_key = {
        tuple(report['vertex_ids_0based']): report
        for report in local_repair_reports
        if isinstance(report, dict) and 'vertex_ids_0based' in report
    }
    two_edge_report_by_path = _report_by_path(two_edge_reports)
    endpoint_bridge_report_by_path = _report_by_path(endpoint_bridge_reports)

    reports = []
    for label, group_id, alternative_id, path in paths:
        reports.append(_classify_human_gap_path(
            mesh=mesh,
            label=label,
            group_id=group_id,
            alternative_id=alternative_id,
            path=tuple(int(value) for value in path),
            edge_by_key=edge_by_key,
            seam_degree=seam_degree,
            seam_adjacency=seam_adjacency,
            component_id_of=component_id_of,
            predicted_key_set=predicted_key_set,
            local_marked=local_marked,
            two_edge_marked=two_edge_marked,
            endpoint_bridge_marked=endpoint_bridge_marked,
            one_edge_report_by_key=one_edge_report_by_key,
            two_edge_report_by_path=two_edge_report_by_path,
            endpoint_bridge_report_by_path=endpoint_bridge_report_by_path,
        ))

    summary = _human_gap_classification_summary(reports)
    if tuple(bool(edge.use_seam) for edge in mesh.edges) != seam_flags_before:
        raise RuntimeError('Human gap classifier modified seam flags.')
    return {
        'summary': summary,
        'paths': reports,
        'rank_among_allowed_candidates_if_available': None,
        'rank_unavailable_reason': 'repair ranking is not exposed without refactoring selection state',
        'read_only': True,
    }


def _classify_human_gap_path(
    *,
    mesh,
    label,
    group_id,
    alternative_id,
    path,
    edge_by_key,
    seam_degree,
    seam_adjacency,
    component_id_of,
    predicted_key_set,
    local_marked,
    two_edge_marked,
    endpoint_bridge_marked,
    one_edge_report_by_key,
    two_edge_report_by_path,
    endpoint_bridge_report_by_path,
):
    edge_keys = _path_edge_keys(path)
    edge_records = [edge_by_key.get(edge_key) for edge_key in edge_keys]
    edge_flags = [None if record is None else bool(record[1].use_seam) for record in edge_records]
    missing_edge_keys = [
        [int(edge_key[0]), int(edge_key[1])]
        for edge_key, record in zip(edge_keys, edge_records)
        if record is None
    ]
    all_edges_exist = all(record is not None for record in edge_records)
    all_marked = bool(all_edges_exist and all(edge_flags))
    partially_marked = bool(all_edges_exist and any(edge_flags) and not all(edge_flags))
    endpoint_vertices = (path[0], path[-1])
    intermediate_vertices = path[1:-1]
    component_ids = [component_id_of.get(vertex) for vertex in path]
    relation = _component_relation(endpoint_vertices, component_id_of, seam_degree)
    total_path_length, endpoint_distance, straightness = _path_geometry(mesh, path)
    tangent_alignments, tangent_available = _classification_tangent_alignments(
        mesh,
        path,
        seam_adjacency,
    )

    phase_2a1_allowed, phase_2a1_reason = _classification_phase_2a1(
        path,
        edge_keys,
        edge_records,
        seam_degree,
        one_edge_report_by_key,
    )
    phase_2b_allowed, phase_2b_reason = _classification_phase_2b(
        path,
        edge_records,
        two_edge_report_by_path,
        seam_degree,
        component_id_of,
        seam_adjacency,
    )
    phase_2b1_allowed, phase_2b1_reason = _classification_phase_2b1(
        path,
        endpoint_bridge_report_by_path,
        edge_records,
        seam_degree,
        component_id_of,
    )
    skipped_due_to_over_cap = phase_2b_reason == 'repair_over_cap' or phase_2b1_reason == 'repair_over_cap'
    candidate_class = _classification_candidate_class(
        path,
        all_edges_exist,
        all_marked,
        phase_2a1_allowed,
        phase_2b_allowed,
        phase_2b_reason,
        phase_2b1_allowed,
        phase_2b1_reason,
        seam_degree,
        intermediate_vertices,
    )
    rejection_reason = _primary_rejection_reason(
        candidate_class,
        all_edges_exist,
        phase_2a1_reason,
        phase_2b_reason,
        phase_2b1_reason,
    )
    return {
        'label': label,
        'preferred_group_id': group_id,
        'alternative_id': alternative_id,
        'path_vertex_ids': [int(vertex) for vertex in path],
        'path_length_edges': len(edge_keys),
        'all_edges_exist_in_blender': bool(all_edges_exist),
        'edge_keys': [[int(a), int(b)] for a, b in edge_keys],
        'blender_edge_indices': [None if record is None else int(record[0]) for record in edge_records],
        'missing_edge_keys': missing_edge_keys,
        'edge_seam_flags_before_classifier': edge_flags,
        'already_all_marked': bool(all_marked),
        'already_partially_marked': bool(partially_marked),
        'marked_by_prediction_if_traceable': bool(any(edge_key in predicted_key_set for edge_key in edge_keys)),
        'marked_by_local_repair_if_traceable': bool(any(edge_key in local_marked for edge_key in edge_keys)),
        'marked_by_two_edge_same_component_if_traceable': bool(
            any(edge_key in two_edge_marked for edge_key in edge_keys)
        ),
        'marked_by_two_edge_endpoint_bridge_if_traceable': bool(
            any(edge_key in endpoint_bridge_marked for edge_key in edge_keys)
        ),
        'vertex_seam_degrees_current': [int(seam_degree.get(vertex, 0)) for vertex in path],
        'endpoint_seam_vertex_flags': [bool(seam_degree.get(vertex, 0) > 0) for vertex in endpoint_vertices],
        'intermediate_seam_vertex_flags': [
            bool(seam_degree.get(vertex, 0) > 0) for vertex in intermediate_vertices
        ],
        'component_ids_current': component_ids,
        'component_relation': relation,
        'total_path_length': total_path_length,
        'endpoint_distance': endpoint_distance,
        'path_straightness': straightness,
        'endpoint_tangent_alignments': tangent_alignments,
        'tangent_available_flags': tangent_available,
        'candidate_class': candidate_class,
        'would_be_allowed_by_phase_2a1': bool(phase_2a1_allowed),
        'phase_2a1_rejection_reason': phase_2a1_reason,
        'would_be_allowed_by_phase_2b_same_component': bool(phase_2b_allowed),
        'phase_2b_rejection_reason': phase_2b_reason,
        'would_be_allowed_by_phase_2b1_endpoint_bridge': bool(phase_2b1_allowed),
        'phase_2b1_rejection_reason': phase_2b1_reason,
        'skipped_only_due_to_over_cap': bool(skipped_due_to_over_cap),
        'rank_among_allowed_candidates_if_available': None,
        'rank_unavailable_reason': 'repair ranking is not exposed without refactoring selection state',
        'rejection_reason': rejection_reason,
    }


def _classification_phase_2a1(path, edge_keys, edge_records, seam_degree, one_edge_report_by_key):
    if len(edge_keys) != 1:
        return False, 'path_length_not_supported'
    report = one_edge_report_by_key.get(edge_keys[0])
    if report is not None:
        reason = report.get('rejection_reason')
        return bool(reason is None or reason == 'repair_over_cap' or report.get('accepted', False)), reason
    if edge_records[0] is None:
        return False, 'edge_not_found'
    u, v = path
    du = seam_degree.get(u, 0)
    dv = seam_degree.get(v, 0)
    if du == 0 or dv == 0:
        return False, 'endpoint_not_seam_vertex'
    if not _is_allowed_missing_edge_degree_pattern(du, dv):
        return False, 'degree_pattern_not_allowed'
    return True, None


def _classification_phase_2b(path, edge_records, report_by_path, seam_degree, component_id_of, seam_adjacency):
    if len(path) != 3:
        return False, 'path_length_not_supported'
    report = report_by_path.get(_canonical_two_edge_path(path))
    if report is not None:
        reason = report.get('rejection_reason')
        return _report_allowed_or_over_cap(report), reason
    if any(record is None for record in edge_records):
        return False, 'edge_not_found'
    u, middle, v = _canonical_two_edge_path(path)
    du = seam_degree.get(u, 0)
    dm = seam_degree.get(middle, 0)
    dv = seam_degree.get(v, 0)
    if du == 0 or dv == 0:
        return False, 'endpoint_not_seam_vertex'
    if dm > 0:
        return False, 'intermediate_not_degree_0'
    if not _is_allowed_two_edge_endpoint_degree_pattern(du, dv):
        return False, 'degree_pattern_not_allowed'
    if component_id_of.get(u) != component_id_of.get(v):
        return False, 'same_component_required'
    distance = _shortest_seam_path_length(seam_adjacency, u, v)
    if distance is None or distance > 3:
        return False, 'same_component_required'
    return True, None


def _classification_phase_2b1(path, report_by_path, edge_records, seam_degree, component_id_of):
    if len(path) != 3:
        return False, 'path_length_not_supported'
    report = report_by_path.get(_canonical_two_edge_path(path))
    if report is not None:
        reason = report.get('rejection_reason')
        return _report_allowed_or_over_cap(report), reason
    if any(record is None for record in edge_records):
        return False, 'edge_not_found'
    u, middle, v = _canonical_two_edge_path(path)
    if seam_degree.get(u, 0) != 1 or seam_degree.get(v, 0) != 1:
        return False, 'endpoint_not_degree_1'
    if seam_degree.get(middle, 0) != 0:
        return False, 'intermediate_not_degree_0'
    if component_id_of.get(u) == component_id_of.get(v):
        return False, 'different_components_required'
    return True, None


def _report_allowed_or_over_cap(report):
    reason = report.get('rejection_reason')
    return bool(
        reason is None
        or reason == 'repair_over_cap'
        or report.get('accepted', False)
        or report.get('accepted_by_normal_rule', False)
        or report.get('accepted_by_target_over_cap_exception', False)
    )


def _classification_candidate_class(
    path,
    all_edges_exist,
    all_marked,
    phase_2a1_allowed,
    phase_2b_allowed,
    phase_2b_reason,
    phase_2b1_allowed,
    phase_2b1_reason,
    seam_degree,
    intermediate_vertices,
):
    if not all_edges_exist:
        return 'non_original_or_missing_blender_edge'
    if all_marked:
        return 'already_marked'
    if len(path) == 2 and phase_2a1_allowed:
        return 'phase_2a1_one_edge_missing_continuity'
    if len(path) == 3 and (phase_2b_allowed or phase_2b_reason == 'repair_over_cap'):
        return 'phase_2b_same_component_two_edge'
    if len(path) == 3 and (phase_2b1_allowed or phase_2b1_reason == 'repair_over_cap'):
        return 'phase_2b1_inter_component_two_edge_endpoint_bridge'
    if len(path) == 4:
        return 'three_edge_local_bridge'
    if any(seam_degree.get(vertex, 0) > 0 for vertex in intermediate_vertices):
        return 'endpoint_to_skeleton_or_near_junction'
    return 'unknown'


def _primary_rejection_reason(candidate_class, all_edges_exist, phase_2a1_reason, phase_2b_reason, phase_2b1_reason):
    if not all_edges_exist:
        return 'edge_not_found'
    if candidate_class == 'phase_2a1_one_edge_missing_continuity':
        return phase_2a1_reason
    if candidate_class == 'phase_2b_same_component_two_edge':
        return phase_2b_reason
    if candidate_class == 'phase_2b1_inter_component_two_edge_endpoint_bridge':
        return phase_2b1_reason
    if candidate_class == 'three_edge_local_bridge':
        return 'path_length_not_supported'
    return phase_2b1_reason or phase_2b_reason or phase_2a1_reason or 'unknown'


def _human_gap_classification_summary(reports):
    count_by_class = _count_by_key(reports, 'candidate_class')
    count_by_relation = _count_by_key(reports, 'component_relation')
    count_by_rejection = _count_by_key(reports, 'rejection_reason')
    total = len(reports)
    summary = {
        'total_paths_classified': total,
        'paths_all_edges_exist_in_blender': sum(1 for report in reports if report['all_edges_exist_in_blender']),
        'paths_missing_blender_edges': sum(1 for report in reports if not report['all_edges_exist_in_blender']),
        'paths_already_all_marked': sum(1 for report in reports if report['already_all_marked']),
        'paths_already_partially_marked': sum(1 for report in reports if report['already_partially_marked']),
        'count_by_candidate_class': count_by_class,
        'count_by_component_relation': count_by_relation,
        'count_by_rejection_reason': count_by_rejection,
        'count_skipped_only_due_to_over_cap': sum(
            1 for report in reports if report['skipped_only_due_to_over_cap']
        ),
        'count_would_be_allowed_by_phase_2a1': sum(
            1 for report in reports if report['would_be_allowed_by_phase_2a1']
        ),
        'count_would_be_allowed_by_phase_2b_same_component': sum(
            1 for report in reports if report['would_be_allowed_by_phase_2b_same_component']
        ),
        'count_would_be_allowed_by_phase_2b1_endpoint_bridge': sum(
            1 for report in reports if report['would_be_allowed_by_phase_2b1_endpoint_bridge']
        ),
    }
    summary['recommended_next_action'] = _recommended_human_gap_next_action(summary)
    return summary


def _recommended_human_gap_next_action(summary):
    total = max(1, int(summary['total_paths_classified']))
    if summary['paths_missing_blender_edges'] > total / 2:
        return 'investigate_non_original_edges'
    if summary['count_by_candidate_class'].get('three_edge_local_bridge', 0) > total / 3:
        return 'add_three_edge_classifier_before_repair'
    if (
        summary['count_would_be_allowed_by_phase_2b_same_component'] > total / 2
        and summary['count_skipped_only_due_to_over_cap'] > 0
    ):
        return 'improve_phase_2b_same_component_ranking'
    if (
        summary['count_would_be_allowed_by_phase_2b1_endpoint_bridge'] > total / 2
        and summary['count_skipped_only_due_to_over_cap'] > 0
    ):
        return 'investigate_phase_2b1_endpoint_bridge_ranking'
    if summary['count_by_candidate_class'].get('endpoint_to_skeleton_or_near_junction', 0) > total / 3:
        return 'investigate_endpoint_to_skeleton'
    return 'no_dominant_class'


def _count_by_key(reports, key):
    counts = {}
    for report in reports:
        value = report.get(key)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _marked_edge_key_set(reports):
    keys = set()
    for report in reports:
        if not isinstance(report, dict):
            continue
        for edge_key in report.get('marked_seam_edges', []):
            if isinstance(edge_key, list) and len(edge_key) == 2:
                keys.add((min(edge_key[0], edge_key[1]), max(edge_key[0], edge_key[1])))
    return keys


def _report_by_path(reports):
    return {
        tuple(report['path_vertex_ids']): report
        for report in reports
        if isinstance(report, dict) and 'path_vertex_ids' in report
    }


def _path_edge_keys(path):
    return [
        (min(path[index], path[index + 1]), max(path[index], path[index + 1]))
        for index in range(len(path) - 1)
    ]


def _component_relation(endpoint_vertices, component_id_of, seam_degree):
    u, v = endpoint_vertices
    u_is_seam = seam_degree.get(u, 0) > 0
    v_is_seam = seam_degree.get(v, 0) > 0
    if not u_is_seam or not v_is_seam:
        return 'endpoint_not_seam'
    component_u = component_id_of.get(u)
    component_v = component_id_of.get(v)
    if component_u is None or component_v is None:
        return 'unknown'
    if component_u == component_v:
        return 'same_component'
    return 'different_components'


def _path_geometry(mesh, path):
    positions = [_vertex_position(mesh, vertex) for vertex in path]
    if any(position is None for position in positions):
        return None, None, None
    total_length = sum(_distance(positions[index], positions[index + 1]) for index in range(len(positions) - 1))
    endpoint_distance = _distance(positions[0], positions[-1])
    straightness = None
    if len(path) == 3:
        first = _normalize(_vector_sub(positions[1], positions[0]))
        second = _normalize(_vector_sub(positions[2], positions[1]))
        if first is not None and second is not None:
            straightness = _dot(first, second)
    return total_length, endpoint_distance, straightness


def _classification_tangent_alignments(mesh, path, seam_adjacency):
    if len(path) != 3:
        return None, [False, False]
    geometry = _two_edge_endpoint_bridge_geometry(mesh, _canonical_two_edge_path(path), seam_adjacency)
    if geometry is None:
        return None, [False, False]
    return (geometry['alignment_u'], geometry['alignment_v']), [True, True]


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
        'endpoint_u_is_seam_vertex': bool(seam_degree.get(u, 0) > 0),
        'endpoint_v_is_seam_vertex': bool(seam_degree.get(v, 0) > 0),
        'degree_pattern': (int(seam_degree.get(u, 0)), int(seam_degree.get(v, 0))),
        'allowed_by_degree_rule': _is_allowed_missing_edge_degree_pattern(
            seam_degree.get(u, 0),
            seam_degree.get(v, 0),
        ),
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
        'blender_local_repair_allowed_candidates_total': (
            result.blender_local_repair_allowed_candidates_total
        ),
        'blender_local_repair_safety_cap': result.blender_local_repair_safety_cap,
        'blender_local_repair_repair_over_cap': result.blender_local_repair_repair_over_cap,
        'blender_local_repair_edges_marked': result.blender_local_repair_edges_marked,
        'blender_local_repair_edges_rejected': result.blender_local_repair_edges_rejected,
        'blender_local_repair_candidate_reports': list(result.blender_local_repair_candidate_reports),
        'human_case_2557_2558_found': result.human_case_2557_2558_found,
        'human_case_2557_2558_edge_exists': result.human_case_2557_2558_edge_exists,
        'human_case_2557_2558_accepted': result.human_case_2557_2558_accepted,
        'human_case_2557_2558_seam_degree_u_before': (
            result.human_case_2557_2558_seam_degree_u_before
        ),
        'human_case_2557_2558_seam_degree_v_before': (
            result.human_case_2557_2558_seam_degree_v_before
        ),
        'human_case_2557_2558_endpoint_u_is_seam_vertex': (
            result.human_case_2557_2558_endpoint_u_is_seam_vertex
        ),
        'human_case_2557_2558_endpoint_v_is_seam_vertex': (
            result.human_case_2557_2558_endpoint_v_is_seam_vertex
        ),
        'human_case_2557_2558_degree_pattern': result.human_case_2557_2558_degree_pattern,
        'human_case_2557_2558_allowed_by_degree_rule': (
            result.human_case_2557_2558_allowed_by_degree_rule
        ),
        'human_case_over_cap_exception_used': result.human_case_over_cap_exception_used,
        'human_case_2557_2558_marked_seam': result.human_case_2557_2558_marked_seam,
        'human_case_2557_2558_rejection_reason': result.human_case_2557_2558_rejection_reason,
        'blender_two_edge_repair_enabled': result.blender_two_edge_repair_enabled,
        'blender_two_edge_repair_candidates_total': result.blender_two_edge_repair_candidates_total,
        'blender_two_edge_repair_allowed_candidates_total': (
            result.blender_two_edge_repair_allowed_candidates_total
        ),
        'blender_two_edge_repair_edges_marked': result.blender_two_edge_repair_edges_marked,
        'blender_two_edge_repair_paths_marked': result.blender_two_edge_repair_paths_marked,
        'blender_two_edge_repair_paths_rejected': result.blender_two_edge_repair_paths_rejected,
        'blender_two_edge_repair_over_cap': result.blender_two_edge_repair_over_cap,
        'blender_two_edge_repair_safety_cap': result.blender_two_edge_repair_safety_cap,
        'blender_two_edge_repair_candidate_reports': list(
            result.blender_two_edge_repair_candidate_reports
        ),
        'blender_two_edge_endpoint_bridge_enabled': result.blender_two_edge_endpoint_bridge_enabled,
        'blender_two_edge_endpoint_bridge_selection_policy': (
            result.blender_two_edge_endpoint_bridge_selection_policy
        ),
        'blender_two_edge_endpoint_bridge_candidates_total': (
            result.blender_two_edge_endpoint_bridge_candidates_total
        ),
        'blender_two_edge_endpoint_bridge_allowed_total': (
            result.blender_two_edge_endpoint_bridge_allowed_total
        ),
        'blender_two_edge_endpoint_bridge_paths_marked': (
            result.blender_two_edge_endpoint_bridge_paths_marked
        ),
        'blender_two_edge_endpoint_bridge_edges_marked': (
            result.blender_two_edge_endpoint_bridge_edges_marked
        ),
        'blender_two_edge_endpoint_bridge_over_cap': result.blender_two_edge_endpoint_bridge_over_cap,
        'blender_two_edge_endpoint_bridge_safety_cap': (
            result.blender_two_edge_endpoint_bridge_safety_cap
        ),
        'blender_two_edge_endpoint_bridge_selected_rank_threshold': (
            result.blender_two_edge_endpoint_bridge_selected_rank_threshold
        ),
        'blender_two_edge_endpoint_bridge_candidate_reports': list(
            result.blender_two_edge_endpoint_bridge_candidate_reports
        ),
        'blender_two_edge_endpoint_bridge_allowed_candidate_reports': list(
            result.blender_two_edge_endpoint_bridge_allowed_candidate_reports
        ),
        'blender_two_edge_endpoint_bridge_human_path_reports': list(
            result.blender_two_edge_endpoint_bridge_human_path_reports
        ),
        'blender_two_edge_endpoint_bridge_human_paths_selected_by_rank': (
            result.blender_two_edge_endpoint_bridge_human_paths_selected_by_rank
        ),
        'blender_two_edge_endpoint_bridge_human_paths_skipped_below_threshold': (
            result.blender_two_edge_endpoint_bridge_human_paths_skipped_below_threshold
        ),
        'target_path_2045_2541_4884_found': result.target_path_2045_2541_4884_found,
        'target_path_2045_2541_4884_allowed': result.target_path_2045_2541_4884_allowed,
        'target_path_2045_2541_4884_marked': result.target_path_2045_2541_4884_marked,
        'target_path_2045_2541_4884_rejection_reason': (
            result.target_path_2045_2541_4884_rejection_reason
        ),
        'target_path_2045_2541_4884_tangent_alignments': (
            result.target_path_2045_2541_4884_tangent_alignments
        ),
        'target_path_2045_2541_4884_straightness': (
            result.target_path_2045_2541_4884_straightness
        ),
        'target_path_2045_2541_4884_accepted_by_normal_rule': (
            result.target_path_2045_2541_4884_accepted_by_normal_rule
        ),
        'target_path_2045_2541_4884_accepted_by_target_over_cap_exception': (
            result.target_path_2045_2541_4884_accepted_by_target_over_cap_exception
        ),
        'target_path_2540_2541_2544_found': result.target_path_2540_2541_2544_found,
        'target_path_2540_2541_2544_allowed': result.target_path_2540_2541_2544_allowed,
        'target_path_2540_2541_2544_marked': result.target_path_2540_2541_2544_marked,
        'target_path_2540_2541_2544_rejection_reason': (
            result.target_path_2540_2541_2544_rejection_reason
        ),
        'target_path_2540_2541_2544_tangent_alignments': (
            result.target_path_2540_2541_2544_tangent_alignments
        ),
        'target_path_2540_2541_2544_straightness': (
            result.target_path_2540_2541_2544_straightness
        ),
        'target_path_2540_2541_2544_accepted_by_normal_rule': (
            result.target_path_2540_2541_2544_accepted_by_normal_rule
        ),
        'target_path_2540_2541_2544_accepted_by_target_over_cap_exception': (
            result.target_path_2540_2541_2544_accepted_by_target_over_cap_exception
        ),
    }
    with open(debug_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    return debug_path


def write_human_gap_classification(json_path, result):
    debug_path = json_path.rsplit('.', 1)[0] + '_human_gap_classification.json'
    payload = result.human_gap_classification or {
        'summary': {
            'total_paths_classified': 0,
            'recommended_next_action': 'no_dominant_class',
        },
        'paths': [],
        'read_only': True,
    }
    with open(debug_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    return debug_path


def build_endpoint_bridge_ranking_debug(result):
    allowed = [
        dict(report)
        for report in result.blender_two_edge_endpoint_bridge_allowed_candidate_reports
    ]
    selected = [report for report in allowed if report.get('selected_for_marking')]
    selected.sort(key=lambda report: report.get('rank') or 10**9)
    top_12 = allowed[:12]
    threshold = result.blender_two_edge_endpoint_bridge_selected_rank_threshold
    human_reports = _endpoint_bridge_debug_human_reports(allowed, threshold)
    old_target_reports = _endpoint_bridge_debug_old_target_reports(allowed, threshold)
    diagnosis = _endpoint_bridge_ranking_diagnosis(
        allowed,
        selected,
        human_reports,
        old_target_reports,
    )
    summary = {
        'selection_policy': result.blender_two_edge_endpoint_bridge_selection_policy,
        'safety_cap': result.blender_two_edge_endpoint_bridge_safety_cap,
        'allowed_total': result.blender_two_edge_endpoint_bridge_allowed_total,
        'selected_total': result.blender_two_edge_endpoint_bridge_paths_marked,
        'over_cap': result.blender_two_edge_endpoint_bridge_over_cap,
        'selected_rank_threshold': threshold,
        'score_tuple_definition': list(ENDPOINT_BRIDGE_SCORE_TUPLE_DEFINITION),
        'human_phase_2b1_total': len(human_reports),
        'human_phase_2b1_selected': sum(
            1 for report in human_reports if report['selected_for_marking']
        ),
        'human_phase_2b1_skipped_below_threshold': sum(
            1 for report in human_reports
            if report['skipped_reason'] == 'over_cap_ranked_below_threshold'
        ),
        'old_validation_targets_selected': sum(
            1 for report in old_target_reports.values()
            if report['selected_for_marking']
        ),
        'old_validation_targets_skipped': sum(
            1 for report in old_target_reports.values()
            if report['skipped_reason'] == 'over_cap_ranked_below_threshold'
        ),
    }
    return {
        'phase_2b1_ranking_summary': summary,
        'full_ranked_allowed_candidates': allowed,
        'top_12_ranked_allowed_candidates': top_12,
        'selected_top_k_candidates': selected,
        'skipped_human_phase_2b1_candidates': [
            report for report in human_reports
            if report['skipped_reason'] == 'over_cap_ranked_below_threshold'
        ],
        'old_validation_target_reports': old_target_reports,
        'ranking_diagnosis': diagnosis,
        'read_only': True,
    }


def write_endpoint_bridge_ranking_debug(json_path, result):
    debug_path = json_path.rsplit('.', 1)[0] + '_endpoint_bridge_ranking_debug.json'
    payload = build_endpoint_bridge_ranking_debug(result)
    with open(debug_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    return debug_path


def format_endpoint_bridge_ranking_debug_summary(payload, debug_path):
    summary = payload['phase_2b1_ranking_summary']
    old_targets = payload['old_validation_target_reports']
    target_bits = []
    for _, path in OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS:
        label = _old_validation_target_label(path)
        report = old_targets[label]
        state = 'selected' if report['selected_for_marking'] else report['skipped_reason']
        target_bits.append(f"{list(path)}={report['rank']} {state}")
    diagnosis = payload['ranking_diagnosis']
    return (
        f"Endpoint bridge ranking debug: {summary['allowed_total']} allowed, "
        f"cap={summary['safety_cap']}, {summary['selected_total']} selected. "
        f"Old targets ranks: {', '.join(target_bits)}. "
        f"Human Phase 2B.1 selected: {summary['human_phase_2b1_selected']}/"
        f"{summary['human_phase_2b1_total']}. "
        f"Recommended next action: {diagnosis['recommended_next_action']}. "
        f"Sidecar: {debug_path}"
    )


def _endpoint_bridge_debug_human_reports(allowed, threshold):
    by_path = {
        tuple(report['path_vertex_ids']): report
        for report in allowed
    }
    reports = []
    for label, group_id, alternative_id, path in HUMAN_GAP_REGRESSION_PATHS:
        if len(path) != 3:
            continue
        report = by_path.get(_canonical_two_edge_path(path))
        if report is None:
            continue
        reports.append({
            'human_path_label': label,
            'preferred_group_id': group_id,
            'alternative_id': alternative_id,
            'path_vertex_ids': list(report['path_vertex_ids']),
            'rank': report.get('rank'),
            'selected_for_marking': bool(report.get('selected_for_marking', False)),
            'marked': bool(report.get('marked', False)),
            'skipped_reason': report.get('skipped_reason'),
            'candidate_score_tuple': report.get('candidate_score_tuple'),
            'total_path_length': report.get('total_path_length'),
            'endpoint_distance': report.get('endpoint_distance'),
            'min_endpoint_tangent_alignment': report.get('min_endpoint_tangent_alignment'),
            'path_straightness': report.get('path_straightness'),
            'rank_delta_from_threshold': _rank_delta(report.get('rank'), threshold),
        })
    return reports


def _endpoint_bridge_debug_old_target_reports(allowed, threshold):
    by_path = {
        tuple(report['path_vertex_ids']): report
        for report in allowed
    }
    selected = [report for report in allowed if report.get('selected_for_marking')]
    threshold_report = _threshold_report(allowed, threshold)
    reports = {}
    for label, path in OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS:
        target_label = _old_validation_target_label(path)
        report = by_path.get(_canonical_two_edge_path(path))
        if report is None:
            reports[target_label] = {
                'found_in_allowed_candidates': False,
                'rank': None,
                'selected_for_marking': False,
                'marked': False,
                'skipped_reason': 'not_found',
                'candidate_score_tuple': None,
                'total_path_length': None,
                'endpoint_distance': None,
                'endpoint_tangent_alignment_u': None,
                'endpoint_tangent_alignment_v': None,
                'min_endpoint_tangent_alignment': None,
                'path_straightness': None,
                'rank_delta_from_threshold': None,
                'primary_penalty_component': 'not_found',
            }
            continue
        reports[target_label] = {
            'found_in_allowed_candidates': True,
            'rank': report.get('rank'),
            'selected_for_marking': bool(report.get('selected_for_marking', False)),
            'marked': bool(report.get('marked', False)),
            'skipped_reason': report.get('skipped_reason'),
            'candidate_score_tuple': report.get('candidate_score_tuple'),
            'total_path_length': report.get('total_path_length'),
            'endpoint_distance': report.get('endpoint_distance'),
            'endpoint_tangent_alignment_u': report.get('endpoint_tangent_alignment_u'),
            'endpoint_tangent_alignment_v': report.get('endpoint_tangent_alignment_v'),
            'min_endpoint_tangent_alignment': report.get('min_endpoint_tangent_alignment'),
            'path_straightness': report.get('path_straightness'),
            'rank_delta_from_threshold': _rank_delta(report.get('rank'), threshold),
            'primary_penalty_component': _primary_penalty_component(
                report,
                threshold_report,
                selected,
            ),
        }
    return reports


def _endpoint_bridge_ranking_diagnosis(allowed, selected, human_reports, old_target_reports):
    selected_labels = sorted({
        label for report in selected
        for label in report.get('human_gap_match_labels', [])
    })
    skipped_labels = sorted({
        report['human_path_label'] for report in human_reports
        if report['skipped_reason'] == 'over_cap_ranked_below_threshold'
    })
    selected_straightness = [
        report['path_straightness'] for report in selected
        if report.get('path_straightness') is not None
    ]
    selected_tangent = [
        report['min_endpoint_tangent_alignment'] for report in selected
        if report.get('min_endpoint_tangent_alignment') is not None
    ]
    median_straightness = _median(selected_straightness)
    median_tangent = _median(selected_tangent)
    worst_selected_length = max(
        (report['total_path_length'] for report in selected if report.get('total_path_length') is not None),
        default=None,
    )
    skipped_human = [
        report for report in human_reports
        if report['skipped_reason'] == 'over_cap_ranked_below_threshold'
    ]
    old_below = {
        label: report['primary_penalty_component']
        for label, report in old_target_reports.items()
        if report['skipped_reason'] == 'over_cap_ranked_below_threshold'
    }
    strong_skipped = [
        report for report in skipped_human
        if (
            median_straightness is not None
            and report.get('path_straightness') is not None
            and report['path_straightness'] >= median_straightness
            and median_tangent is not None
            and report.get('min_endpoint_tangent_alignment') is not None
            and report['min_endpoint_tangent_alignment'] >= median_tangent
        )
    ]
    length_first_bias = bool(
        old_below
        and all(value in ('total_path_length', 'endpoint_distance') for value in old_below.values())
    ) or bool(strong_skipped and any(
        report['total_path_length'] is not None
        and worst_selected_length is not None
        and report['total_path_length'] <= worst_selected_length
        for report in strong_skipped
    ))
    if length_first_bias:
        recommended = 'revise_phase_2b1_ranking_formula'
    elif selected_labels and len(selected_labels) >= max(1, len(human_reports) // 2):
        recommended = 'inspect_selected_candidates_visually'
    elif allowed:
        recommended = 'keep_ranking_collect_visual_feedback'
    else:
        recommended = 'no_action'
    return {
        'selected_human_match_labels': selected_labels,
        'skipped_human_match_labels': skipped_labels,
        'skipped_human_paths_above_median_selected_straightness': [
            report['human_path_label'] for report in skipped_human
            if (
                median_straightness is not None
                and report.get('path_straightness') is not None
                and report['path_straightness'] >= median_straightness
            )
        ],
        'skipped_human_paths_above_median_selected_tangent': [
            report['human_path_label'] for report in skipped_human
            if (
                median_tangent is not None
                and report.get('min_endpoint_tangent_alignment') is not None
                and report['min_endpoint_tangent_alignment'] >= median_tangent
            )
        ],
        'skipped_human_paths_shorter_than_worst_selected': [
            report['human_path_label'] for report in skipped_human
            if (
                worst_selected_length is not None
                and report.get('total_path_length') is not None
                and report['total_path_length'] <= worst_selected_length
            )
        ],
        'old_targets_ranked_below_threshold_because': old_below,
        'length_first_bias_suspected': bool(length_first_bias),
        'recommended_next_action': recommended,
    }


def _old_validation_target_label(path):
    for label, target_path in OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS:
        if _canonical_two_edge_path(path) == _canonical_two_edge_path(target_path):
            return label
    return None


def _threshold_report(allowed, threshold):
    if threshold is None or threshold < 1 or threshold > len(allowed):
        return None
    return allowed[threshold - 1]


def _primary_penalty_component(report, threshold_report, selected):
    if report.get('conflict_reason'):
        return 'conflict'
    if threshold_report is None or report.get('candidate_score_tuple') is None:
        return 'unknown'
    score = report['candidate_score_tuple']
    threshold_score = threshold_report.get('candidate_score_tuple')
    if threshold_score is None:
        return 'unknown'
    labels = (
        'total_path_length',
        'endpoint_distance',
        'tangent_alignment',
        'path_straightness',
        'tie_break',
    )
    for index, label in enumerate(labels):
        if score[index] > threshold_score[index]:
            return label
        if score[index] < threshold_score[index]:
            return 'unknown'
    return 'unknown'


def _rank_delta(rank, threshold):
    if rank is None or threshold is None:
        return None
    return int(rank) - int(threshold)


def _median(values):
    if not values:
        return None
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


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
    degree = result.human_case_2557_2558_degree_pattern
    degree_suffix = f', degree={degree}' if degree is not None else ''
    if result.human_case_2557_2558_marked_seam and result.human_case_over_cap_exception_used:
        human_status = f'marked by over-cap human-case exception{degree_suffix}'
    elif result.human_case_2557_2558_marked_seam:
        human_status = f'marked{degree_suffix}'
    elif result.human_case_2557_2558_found:
        human_status = (
            f"rejected:{result.human_case_2557_2558_rejection_reason}{degree_suffix}"
        )
    else:
        human_status = 'not found'
    target_a_status = _format_two_edge_target_status(
        result.target_path_2045_2541_4884_found,
        result.target_path_2045_2541_4884_marked,
        result.target_path_2045_2541_4884_rejection_reason,
        result.target_path_2045_2541_4884_tangent_alignments,
        result.target_path_2045_2541_4884_straightness,
        result.target_path_2045_2541_4884_accepted_by_target_over_cap_exception,
    )
    target_b_status = _format_two_edge_target_status(
        result.target_path_2540_2541_2544_found,
        result.target_path_2540_2541_2544_marked,
        result.target_path_2540_2541_2544_rejection_reason,
        result.target_path_2540_2541_2544_tangent_alignments,
        result.target_path_2540_2541_2544_straightness,
        result.target_path_2540_2541_2544_accepted_by_target_over_cap_exception,
    )
    classification_summary = (result.human_gap_classification or {}).get('summary', {})
    classified = classification_summary.get('total_paths_classified', 0)
    editable = classification_summary.get('paths_all_edges_exist_in_blender', 0)
    class_counts = classification_summary.get('count_by_candidate_class', {})
    recommended = classification_summary.get('recommended_next_action', 'no_dominant_class')
    return (
        f'Marked {result.applied} seam edges. '
        f'Ignored {result.ignored_non_original} triangulation-only edges. '
        f'Skipped {result.duplicates_skipped} duplicates. '
        f'Bridge debug: {result.accepted_bridge_edges_present_in_json} accepted in JSON, '
        f'{result.accepted_bridge_edges_applied} applied, '
        f'{result.accepted_bridge_edges_ignored_non_original} ignored as non-original.'
        f' Local repair: {result.blender_local_repair_edges_marked} marked, '
        f'{result.blender_local_repair_edges_rejected} rejected, '
        f'allowed={result.blender_local_repair_allowed_candidates_total}, '
        f'over_cap={str(result.blender_local_repair_repair_over_cap).lower()}. '
        f'Human case [2557,2558]: {human_status}.'
        f' Two-edge repair: {result.blender_two_edge_repair_paths_marked} paths marked, '
        f'{result.blender_two_edge_repair_edges_marked} edges marked, '
        f'allowed={result.blender_two_edge_repair_allowed_candidates_total}, '
        f'over_cap={str(result.blender_two_edge_repair_over_cap).lower()}. '
        f'Two-edge endpoint bridge: {result.blender_two_edge_endpoint_bridge_paths_marked} '
        f'paths marked, {result.blender_two_edge_endpoint_bridge_edges_marked} edges marked, '
        f'allowed={result.blender_two_edge_endpoint_bridge_allowed_total}, '
        f'over_cap={str(result.blender_two_edge_endpoint_bridge_over_cap).lower()}, '
        f'policy={result.blender_two_edge_endpoint_bridge_selection_policy}. '
        f'Human Phase 2B.1 paths selected: '
        f'{result.blender_two_edge_endpoint_bridge_human_paths_selected_by_rank}/'
        f'{len(result.blender_two_edge_endpoint_bridge_human_path_reports)}. '
        f'Human Phase 2B.1 paths skipped below rank threshold: '
        f'{result.blender_two_edge_endpoint_bridge_human_paths_skipped_below_threshold}/'
        f'{len(result.blender_two_edge_endpoint_bridge_human_path_reports)}. '
        f'Target [2045,2541,4884]: {target_a_status}. '
        f'Target [2540,2541,2544]: {target_b_status}.'
        f' Human gap classifier: {classified} paths classified, {editable} editable, '
        f'{class_counts}. Recommended next action: {recommended}.'
        f'{trace_suffix}'
    )


def _format_two_edge_target_status(
    found,
    marked,
    rejection_reason,
    tangent_alignments=None,
    straightness=None,
    accepted_by_target_over_cap_exception=False,
):
    if marked:
        if accepted_by_target_over_cap_exception:
            return 'marked by over-cap target exception'
        if tangent_alignments is not None and straightness is not None:
            return f'marked, alignments={tangent_alignments}, straightness={straightness}'
        return 'marked'
    if found:
        return f'rejected:{rejection_reason}'
    return 'not found'


def _is_vertex_pair(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and type(value[0]) is int
        and type(value[1]) is int
        and value[0] >= 0
        and value[1] >= 0
    )
