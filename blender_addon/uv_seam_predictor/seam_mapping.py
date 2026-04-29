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
    blender_two_edge_endpoint_bridge_selection_policy: str = 'top_k_ranked_continuity_tier_v2'
    blender_two_edge_endpoint_bridge_candidates_total: int = 0
    blender_two_edge_endpoint_bridge_raw_allowed_total: int = 0
    blender_two_edge_endpoint_bridge_deduplicated_allowed_total: int = 0
    blender_two_edge_endpoint_bridge_allowed_total: int = 0
    blender_two_edge_endpoint_bridge_paths_marked: int = 0
    blender_two_edge_endpoint_bridge_edges_marked: int = 0
    blender_two_edge_endpoint_bridge_over_cap: bool = False
    blender_two_edge_endpoint_bridge_safety_cap: int = 9
    blender_two_edge_endpoint_bridge_selected_rank_threshold: int | None = None
    blender_two_edge_endpoint_bridge_duplicate_endpoint_pairs_suppressed: int = 0
    blender_two_edge_endpoint_bridge_candidate_reports: tuple = ()
    blender_two_edge_endpoint_bridge_allowed_candidate_reports: tuple = ()
    blender_two_edge_endpoint_bridge_human_paths_selected_by_rank: int = 0
    blender_two_edge_endpoint_bridge_human_paths_skipped_below_threshold: int = 0
    blender_two_edge_endpoint_bridge_human_path_reports: tuple = ()
    blender_two_edge_endpoint_bridge_added_candidate_due_to_cap_increase: bool = False
    blender_two_edge_endpoint_bridge_previous_rank_9_selected: bool = False
    blender_two_edge_endpoint_bridge_selected_rank_9_candidate: dict | None = None
    blender_curved_two_edge_endpoint_bridge_enabled: bool = False
    blender_curved_two_edge_endpoint_bridge_candidates_total: int = 0
    blender_curved_two_edge_endpoint_bridge_eligible_total: int = 0
    blender_curved_two_edge_endpoint_bridge_paths_marked: int = 0
    blender_curved_two_edge_endpoint_bridge_edges_marked: int = 0
    blender_curved_two_edge_endpoint_bridge_over_cap: bool = False
    blender_curved_two_edge_endpoint_bridge_safety_cap: int = 6
    blender_curved_two_edge_endpoint_bridge_candidate_reports: tuple = ()
    selected_curved_two_edge_endpoint_bridge_reports: tuple = ()
    rejected_curved_two_edge_endpoint_bridge_reports: tuple = ()
    probabilities_used_for_curved_repair: bool = False
    phase2j_curved_repair_uses_hardcoded_paths: bool = False
    phase2j_curved_repair_uses_probabilities: bool = False
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
    residual_gap_phase2e_debug: dict | None = None
    general_residual_candidates_phase2h: dict | None = None
    production_hardcoded_path_exceptions_enabled: bool = False
    production_hardcoded_path_exceptions_removed: bool = True
    diagnostic_path_labels_read_only: bool = True
    unified_local_continuity_simulation_phase2h_r: dict | None = None
    phase2h_r3_visual_review: dict | None = None
    phase2j_r_small_gap_rule_simulation: dict | None = None


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
    residual_gap_phase2e_debug = classify_residual_gap_phase2e(
        mesh,
        predicted_keys=applied_keys,
        local_repair_reports=repair['candidate_reports'],
        two_edge_reports=two_edge_repair['candidate_reports'],
        endpoint_bridge_reports=endpoint_bridge_repair['candidate_reports'],
    )
    general_residual_candidates_phase2h = collect_general_residual_candidates_phase2h(
        mesh,
        predicted_keys=applied_keys,
        local_repair_reports=repair['candidate_reports'],
        two_edge_reports=two_edge_repair['candidate_reports'],
        endpoint_bridge_reports=endpoint_bridge_repair['candidate_reports'],
        residual_payload=residual_gap_phase2e_debug,
    )
    unified_local_continuity_simulation_phase2h_r = simulate_unified_local_continuity_phase2h_r(
        mesh,
        predicted_keys=applied_keys,
        local_repair_reports=repair['candidate_reports'],
        two_edge_reports=two_edge_repair['candidate_reports'],
        endpoint_bridge_reports=endpoint_bridge_repair['candidate_reports'],
        residual_payload=residual_gap_phase2e_debug,
    )
    phase2h_r3_visual_review = build_phase2h_r3_visual_review(
        unified_local_continuity_simulation_phase2h_r,
    )
    phase2j_r_small_gap_rule_simulation = simulate_phase2j_r_small_gap_rule(
        mesh,
        predicted_keys=applied_keys,
        local_repair_reports=repair['candidate_reports'],
        two_edge_reports=two_edge_repair['candidate_reports'],
        endpoint_bridge_reports=endpoint_bridge_repair['candidate_reports'],
        residual_payload=residual_gap_phase2e_debug,
    )
    curved_endpoint_bridge_repair = apply_curved_two_edge_endpoint_bridge_repair(
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
        blender_two_edge_endpoint_bridge_raw_allowed_total=int(endpoint_bridge_repair[
            'raw_allowed_total'
        ]),
        blender_two_edge_endpoint_bridge_deduplicated_allowed_total=int(endpoint_bridge_repair[
            'deduplicated_allowed_total'
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
        blender_two_edge_endpoint_bridge_duplicate_endpoint_pairs_suppressed=int(
            endpoint_bridge_repair['duplicate_endpoint_pairs_suppressed']
        ),
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
        blender_two_edge_endpoint_bridge_added_candidate_due_to_cap_increase=bool(
            endpoint_bridge_repair['added_candidate_due_to_cap_increase']
        ),
        blender_two_edge_endpoint_bridge_previous_rank_9_selected=bool(
            endpoint_bridge_repair['previous_rank_9_selected']
        ),
        blender_two_edge_endpoint_bridge_selected_rank_9_candidate=endpoint_bridge_repair[
            'selected_rank_9_candidate'
        ],
        blender_curved_two_edge_endpoint_bridge_enabled=bool(curved_endpoint_bridge_repair['enabled']),
        blender_curved_two_edge_endpoint_bridge_candidates_total=int(
            curved_endpoint_bridge_repair['candidates_total']
        ),
        blender_curved_two_edge_endpoint_bridge_eligible_total=int(
            curved_endpoint_bridge_repair['eligible_total']
        ),
        blender_curved_two_edge_endpoint_bridge_paths_marked=int(
            curved_endpoint_bridge_repair['paths_marked']
        ),
        blender_curved_two_edge_endpoint_bridge_edges_marked=int(
            curved_endpoint_bridge_repair['edges_marked']
        ),
        blender_curved_two_edge_endpoint_bridge_over_cap=bool(
            curved_endpoint_bridge_repair['over_cap']
        ),
        blender_curved_two_edge_endpoint_bridge_safety_cap=int(
            curved_endpoint_bridge_repair['safety_cap']
        ),
        blender_curved_two_edge_endpoint_bridge_candidate_reports=tuple(
            curved_endpoint_bridge_repair['candidate_reports']
        ),
        selected_curved_two_edge_endpoint_bridge_reports=tuple(
            curved_endpoint_bridge_repair['selected_reports']
        ),
        rejected_curved_two_edge_endpoint_bridge_reports=tuple(
            curved_endpoint_bridge_repair['rejected_reports']
        ),
        probabilities_used_for_curved_repair=False,
        phase2j_curved_repair_uses_hardcoded_paths=False,
        phase2j_curved_repair_uses_probabilities=False,
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
        residual_gap_phase2e_debug=residual_gap_phase2e_debug,
        general_residual_candidates_phase2h=general_residual_candidates_phase2h,
        production_hardcoded_path_exceptions_enabled=False,
        production_hardcoded_path_exceptions_removed=True,
        diagnostic_path_labels_read_only=True,
        unified_local_continuity_simulation_phase2h_r=unified_local_continuity_simulation_phase2h_r,
        phase2h_r3_visual_review=phase2h_r3_visual_review,
        phase2j_r_small_gap_rule_simulation=phase2j_r_small_gap_rule_simulation,
    )


def apply_missing_edge_continuity_repair(mesh, enabled=True, max_repair_edges=32):
    human_case = DIAGNOSTIC_HUMAN_CASE_2557_2558
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

    target_keys = _diagnostic_old_endpoint_bridge_target_keys()
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

    allowed_count = len(allowed_paths)
    over_cap = allowed_count > int(max_repair_paths)
    paths_to_mark = set()
    if not over_cap:
        paths_to_mark = set(allowed_paths)

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
    max_repair_paths=9,
):
    edge_items, edge_by_key, adjacency = _mesh_edge_lookup(mesh)
    target_keys = _diagnostic_old_endpoint_bridge_target_keys()
    if not enabled:
        return _two_edge_endpoint_bridge_result(
            enabled=False,
            candidate_reports=(),
            allowed_reports=(),
            deduplicated_allowed_reports=(),
            safety_cap=max_repair_paths,
            over_cap=False,
            selected_rank_threshold=None,
            duplicate_endpoint_pairs_suppressed=0,
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

    allowed_reports = [
        report for report in candidate_reports
        if report['rejection_reason'] is None
    ]
    v1_ranked_reports = sorted(allowed_reports, key=_two_edge_endpoint_bridge_sort_key_v1)
    _annotate_endpoint_bridge_v1_reports(v1_ranked_reports)
    allowed_reports.sort(key=_two_edge_endpoint_bridge_sort_key_v2)
    _annotate_ranked_endpoint_bridge_allowed_reports(allowed_reports)
    deduplicated_allowed_reports = _deduplicate_endpoint_bridge_reports(allowed_reports)
    duplicate_suppressed = len(allowed_reports) - len(deduplicated_allowed_reports)
    over_cap = len(deduplicated_allowed_reports) > int(max_repair_paths)
    reports_to_mark = list(deduplicated_allowed_reports[:int(max_repair_paths)])
    selected_rank_threshold = None
    if reports_to_mark:
        selected_rank_threshold = max(report['rank_v2'] for report in reports_to_mark)
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
        for report in deduplicated_allowed_reports:
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
        deduplicated_allowed_reports=tuple(deduplicated_allowed_reports),
        safety_cap=max_repair_paths,
        over_cap=over_cap,
        selected_rank_threshold=selected_rank_threshold,
        duplicate_endpoint_pairs_suppressed=duplicate_suppressed,
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
    deduplicated_allowed_reports,
    safety_cap,
    over_cap,
    selected_rank_threshold,
    duplicate_endpoint_pairs_suppressed,
    edges_marked,
    paths_marked,
    edge_by_key,
    target_keys,
):
    allowed_candidate_reports = tuple(_allowed_endpoint_bridge_report(report) for report in allowed_reports)
    human_path_reports = _endpoint_bridge_human_path_reports(allowed_reports)
    selected_rank_9_candidate = next(
        (
            report for report in allowed_candidate_reports
            if report.get('rank_v2') == 9 and report.get('selected_for_marking')
        ),
        None,
    )
    result = {
        'enabled': bool(enabled),
        'selection_policy': 'top_k_ranked_continuity_tier_v2',
        'candidates_total': len(candidate_reports),
        'raw_allowed_total': len(allowed_reports),
        'deduplicated_allowed_total': len(deduplicated_allowed_reports),
        'allowed_total': len(allowed_reports),
        'paths_marked': int(paths_marked),
        'edges_marked': int(edges_marked),
        'over_cap': bool(over_cap),
        'safety_cap': int(safety_cap),
        'selected_rank_threshold': selected_rank_threshold,
        'duplicate_endpoint_pairs_suppressed': int(duplicate_endpoint_pairs_suppressed),
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
        'added_candidate_due_to_cap_increase': bool(selected_rank_9_candidate),
        'previous_rank_9_selected': bool(selected_rank_9_candidate),
        'selected_rank_9_candidate': selected_rank_9_candidate,
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


def apply_curved_two_edge_endpoint_bridge_repair(mesh, enabled=True, max_repair_paths=6):
    edge_items, edge_by_key, adjacency = _mesh_edge_lookup(mesh)
    if not enabled:
        return {
            'enabled': False,
            'candidates_total': 0,
            'eligible_total': 0,
            'paths_marked': 0,
            'edges_marked': 0,
            'over_cap': False,
            'safety_cap': int(max_repair_paths),
            'candidate_reports': tuple(),
            'selected_reports': tuple(),
            'rejected_reports': tuple(),
        }

    seam_degree, seam_adjacency = _seam_topology_from_mesh_edges(edge_items)
    component_id_of = _seam_component_ids(seam_adjacency)
    bbox_diagonal = _mesh_bbox_diagonal(mesh)
    candidate_reports = []
    report_by_path = {}

    for middle in sorted(adjacency):
        neighbors = sorted(adjacency[middle])
        for left_index in range(len(neighbors)):
            for right_index in range(left_index + 1, len(neighbors)):
                path = _canonical_two_edge_path((
                    neighbors[left_index],
                    middle,
                    neighbors[right_index],
                ))
                if path[1] != middle:
                    continue
                report = _curved_two_edge_endpoint_bridge_report(
                    mesh=mesh,
                    path=path,
                    edge_by_key=edge_by_key,
                    seam_degree=seam_degree,
                    seam_adjacency=seam_adjacency,
                    component_id_of=component_id_of,
                    bbox_diagonal=bbox_diagonal,
                )
                candidate_reports.append(report)
                report_by_path[path] = report

    raw_eligible = [
        report for report in candidate_reports
        if report['rejection_reason'] is None
    ]
    raw_eligible.sort(key=_curved_endpoint_bridge_sort_key)
    for rank, report in enumerate(raw_eligible, start=1):
        report['rank_if_eligible'] = rank

    deduplicated = []
    seen_endpoint_pairs = set()
    for report in raw_eligible:
        pair = tuple(report['duplicate_endpoint_pair_key'])
        if pair in seen_endpoint_pairs:
            report['rejection_reason'] = 'duplicate_endpoint_pair_suppressed'
            report['selected_for_marking'] = False
            report['marked'] = False
            continue
        seen_endpoint_pairs.add(pair)
        deduplicated.append(report)

    over_cap = len(deduplicated) > int(max_repair_paths)
    selected_reports = list(deduplicated[:int(max_repair_paths)])
    selected_ids = {id(report) for report in selected_reports}
    if over_cap:
        for report in deduplicated:
            if id(report) not in selected_ids:
                report['rejection_reason'] = 'repair_over_cap'

    marked_edge_keys = set()
    for report in selected_reports:
        report['selected_for_marking'] = True
        marked_for_path = []
        for edge_key in [tuple(edge_key) for edge_key in report['path_edge_keys']]:
            edge = edge_by_key.get(edge_key, (None, None))[1]
            if edge is None:
                continue
            edge.use_seam = True
            marked_edge_keys.add(edge_key)
            marked_for_path.append([int(edge_key[0]), int(edge_key[1])])
        report['marked'] = True
        report['accepted'] = True
        report['marked_edge_count'] = len(marked_for_path)
        report['marked_seam_edges'] = marked_for_path
        report['rejection_reason'] = None
        report['accepted_by_normal_rule'] = True

    rejected_reports = [
        report for report in candidate_reports
        if not report.get('marked', False)
    ]
    return {
        'enabled': True,
        'candidates_total': len(candidate_reports),
        'eligible_total': len(deduplicated),
        'paths_marked': sum(1 for report in selected_reports if report.get('marked', False)),
        'edges_marked': len(marked_edge_keys),
        'over_cap': bool(over_cap),
        'safety_cap': int(max_repair_paths),
        'candidate_reports': tuple(candidate_reports),
        'selected_reports': tuple(selected_reports),
        'rejected_reports': tuple(rejected_reports),
    }


def _curved_two_edge_endpoint_bridge_report(
    *,
    mesh,
    path,
    edge_by_key,
    seam_degree,
    seam_adjacency,
    component_id_of,
    bbox_diagonal,
):
    u, middle, v = path
    edge_keys = _two_edge_path_edge_keys(path)
    edge_records = [edge_by_key.get(edge_key) for edge_key in edge_keys]
    edge_indices = [None if record is None else int(record[0]) for record in edge_records]
    edge_flags = [None if record is None else bool(record[1].use_seam) for record in edge_records]
    all_edges_exist = all(record is not None for record in edge_records)
    degree_pattern = [int(seam_degree.get(vertex, 0)) for vertex in path]
    endpoint_seam_flags = [degree_pattern[0] > 0, degree_pattern[-1] > 0]
    intermediate_seam_flag = degree_pattern[1] > 0
    component_ids = [component_id_of.get(vertex) for vertex in path]
    component_relation = _phase2h_component_relation((u, v), component_id_of, seam_degree)
    would_create_loop = component_relation == 'same_component'
    seam_distance = None
    if would_create_loop:
        seam_distance = _shortest_seam_path_length(seam_adjacency, u, v)
    loop_risk = _phase2h_loop_risk(component_relation, seam_distance)
    topology_risk = (
        'accepted_pattern'
        if degree_pattern == [1, 0, 1] and component_relation == 'different_components'
        else _phase2h_topology_risk('two_edge_same_component_local_closure' if would_create_loop else 'unsupported_or_unknown')
    )
    geometry = None
    total_path_length = None
    endpoint_distance = None
    normalized_total = None
    normalized_endpoint = None
    path_straightness = None
    alignments = None
    min_alignment = None
    segment_length_ratio = None
    turn_angle = None

    if all_edges_exist:
        geometry = _two_edge_endpoint_bridge_geometry(mesh, path, seam_adjacency)
    if geometry is not None:
        total_path_length = geometry['total_path_length']
        endpoint_distance = geometry['endpoint_distance']
        normalized_total = _safe_ratio(total_path_length, bbox_diagonal)
        normalized_endpoint = _safe_ratio(endpoint_distance, bbox_diagonal)
        path_straightness = geometry['path_straightness']
        alignments = (geometry['alignment_u'], geometry['alignment_v'])
        min_alignment = min(alignments)
        segment_length_ratio = _safe_ratio(endpoint_distance, total_path_length)
        turn_angle = _turn_angle_from_straightness(path_straightness)

    rejection_reason = _curved_endpoint_bridge_rejection_reason(
        all_edges_exist=all_edges_exist,
        edge_flags=edge_flags,
        degree_pattern=degree_pattern,
        component_relation=component_relation,
        loop_risk=loop_risk,
        topology_risk=topology_risk,
        geometry=geometry,
        normalized_total=normalized_total,
        min_alignment=min_alignment,
        path_straightness=path_straightness,
        segment_length_ratio=segment_length_ratio,
    )
    return {
        'path_vertex_ids': [int(u), int(middle), int(v)],
        'path_edge_keys': [[int(a), int(b)] for a, b in edge_keys],
        'path_edge_indices_blender': edge_indices,
        'blender_edge_indices': edge_indices,
        'degree_pattern': degree_pattern,
        'endpoint_seam_vertex_flags': endpoint_seam_flags,
        'intermediate_seam_vertex_flag': bool(intermediate_seam_flag),
        'component_ids': component_ids,
        'component_ids_before': [component_ids[0], component_ids[-1]],
        'component_relation': component_relation,
        'would_create_loop': bool(would_create_loop),
        'loop_risk': loop_risk,
        'topology_risk': topology_risk,
        'total_path_length': total_path_length,
        'endpoint_distance': endpoint_distance,
        'normalized_total_path_length': normalized_total,
        'normalized_endpoint_distance': normalized_endpoint,
        'path_straightness': path_straightness,
        'endpoint_tangent_alignments': alignments,
        'endpoint_tangent_alignment_u': None if alignments is None else alignments[0],
        'endpoint_tangent_alignment_v': None if alignments is None else alignments[1],
        'min_endpoint_tangent_alignment': min_alignment,
        'segment_length_ratio': segment_length_ratio,
        'turn_angle_degrees': turn_angle,
        'bend_angle_degrees': turn_angle,
        'duplicate_endpoint_pair_key': sorted([int(u), int(v)]),
        'endpoint_pair_key': sorted([int(u), int(v)]),
        'selected_for_marking': False,
        'marked': False,
        'accepted': False,
        'accepted_by_normal_rule': False,
        'marked_edge_count': 0,
        'marked_seam_edges': [],
        'rejection_reason': rejection_reason,
        'rank_if_eligible': None,
        'diagnostic_residual_labels_if_any': [],
        'diagnostic_labels_used_for_selection': False,
        'probabilities_used_for_selection': False,
    }


def _curved_endpoint_bridge_rejection_reason(
    *,
    all_edges_exist,
    edge_flags,
    degree_pattern,
    component_relation,
    loop_risk,
    topology_risk,
    geometry,
    normalized_total,
    min_alignment,
    path_straightness,
    segment_length_ratio,
):
    if not all_edges_exist:
        return 'edge_not_found'
    if any(flag is True for flag in edge_flags):
        return 'edge_already_seam'
    if degree_pattern != [1, 0, 1]:
        if degree_pattern[1] != 0:
            return 'intermediate_not_degree_0'
        return 'endpoint_not_degree_1'
    if component_relation != 'different_components':
        return 'same_component_not_endpoint_bridge'
    if loop_risk != 'none':
        return 'would_create_loop'
    if topology_risk != 'accepted_pattern':
        return 'topology_not_accepted_pattern'
    if geometry is None:
        return 'tangent_unavailable'
    if normalized_total is None or normalized_total > 0.015:
        return 'path_too_long'
    if min_alignment is None or min_alignment < 0.75:
        return 'tangent_alignment_below_threshold'
    if path_straightness is None or path_straightness >= 0.50:
        return 'not_curved_low_straightness'
    if segment_length_ratio is not None and segment_length_ratio < 0.70:
        return 'segment_length_ratio_below_threshold'
    return None


def _curved_endpoint_bridge_sort_key(report):
    return (
        -float(report['min_endpoint_tangent_alignment']),
        float(report['normalized_total_path_length']),
        -float(report['segment_length_ratio']),
        -float(report['path_straightness']),
        list(report['path_vertex_ids']),
    )


def _turn_angle_from_straightness(straightness):
    if straightness is None:
        return None
    clamped = max(-1.0, min(1.0, float(straightness)))
    return math.degrees(math.acos(clamped))


def _annotate_endpoint_bridge_v1_reports(allowed_reports):
    for rank, report in enumerate(allowed_reports, start=1):
        report['rank_v1_length_first'] = rank
        report['candidate_score_tuple_v1_length_first'] = _endpoint_bridge_score_tuple_v1(report)


def _annotate_ranked_endpoint_bridge_allowed_reports(allowed_reports):
    for rank, report in enumerate(allowed_reports, start=1):
        score = _endpoint_bridge_score_tuple_v2(report)
        report['rank_v2'] = rank
        report['rank'] = rank
        report['rank_delta_v2_minus_v1'] = rank - report['rank_v1_length_first']
        report['candidate_score_tuple_v2'] = score
        report['candidate_score_tuple'] = score
        report['continuity_tier'] = _endpoint_bridge_continuity_tier(report)
        report['q_floor'] = min(
            report['min_endpoint_tangent_alignment'],
            report['path_straightness'],
        )
        report['q_sum'] = (
            report['min_endpoint_tangent_alignment']
            + report['path_straightness']
        )
        report['endpoint_pair_key'] = sorted([report['path_vertex_ids'][0], report['path_vertex_ids'][2]])
        report['duplicate_endpoint_pair_suppressed'] = False
        report['selected_for_marking'] = False
        report['marked'] = False
        report['skipped_reason'] = None
        report['conflict_reason'] = None
        report['human_gap_match_labels'] = _human_gap_labels_for_path(report['path_vertex_ids'])
        report['old_validation_target_match_label'] = _old_validation_target_label(report['path_vertex_ids'])


def _endpoint_bridge_score_tuple_v1(report):
    return [
        report['total_path_length'],
        report['endpoint_distance'],
        -report['min_endpoint_tangent_alignment'],
        -report['path_straightness'],
        list(report['path_vertex_ids']),
    ]


def _endpoint_bridge_score_tuple_v2(report):
    t = report['min_endpoint_tangent_alignment']
    s = report['path_straightness']
    q_floor = min(t, s)
    q_sum = t + s
    return [
        _endpoint_bridge_continuity_tier(report),
        -q_floor,
        -q_sum,
        report['total_path_length'],
        report['endpoint_distance'],
        list(report['path_vertex_ids']),
    ]


def _endpoint_bridge_continuity_tier(report):
    t = report['min_endpoint_tangent_alignment']
    s = report['path_straightness']
    if s >= 0.85 and t >= 0.85:
        return 0
    if s >= 0.70 and t >= 0.70:
        return 1
    if s >= 0.50 and t >= 0.75:
        return 2
    return 3


def _deduplicate_endpoint_bridge_reports(allowed_reports):
    selected = []
    seen_endpoint_pairs = set()
    for report in allowed_reports:
        key = tuple(report['endpoint_pair_key'])
        if key in seen_endpoint_pairs:
            report['duplicate_endpoint_pair_suppressed'] = True
            report['skipped_reason'] = 'duplicate_endpoint_pair_suppressed'
            continue
        seen_endpoint_pairs.add(key)
        selected.append(report)
    return selected


def _allowed_endpoint_bridge_report(report):
    return {
        'rank': report.get('rank_v2'),
        'rank_v1_length_first': report.get('rank_v1_length_first'),
        'candidate_score_tuple_v1_length_first': report.get('candidate_score_tuple_v1_length_first'),
        'rank_v2': report.get('rank_v2'),
        'rank_delta_v2_minus_v1': report.get('rank_delta_v2_minus_v1'),
        'candidate_score_tuple_v2': report.get('candidate_score_tuple_v2'),
        'candidate_score_tuple': report.get('candidate_score_tuple_v2'),
        'continuity_tier': report.get('continuity_tier'),
        'q_floor': report.get('q_floor'),
        'q_sum': report.get('q_sum'),
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
        'endpoint_pair_key': list(report.get('endpoint_pair_key', [])),
        'duplicate_endpoint_pair_suppressed': bool(
            report.get('duplicate_endpoint_pair_suppressed', False)
        ),
        'human_gap_match_labels': list(report.get('human_gap_match_labels', [])),
        'old_validation_target_match_label': report.get('old_validation_target_match_label'),
        'conflict_reason': report.get('conflict_reason'),
    }


def _endpoint_bridge_human_path_reports(allowed_reports):
    report_by_path = {
        tuple(report['path_vertex_ids']): report
        for report in allowed_reports
    }
    human_reports = []
    for label, _, _, path in DIAGNOSTIC_HUMAN_GAP_REGRESSION_PATHS:
        if len(path) != 3:
            continue
        canonical = _canonical_two_edge_path(path)
        report = report_by_path.get(canonical)
        if report is None:
            continue
        human_reports.append({
            'human_path_label': label,
            'allowed_rank': report.get('rank_v2'),
            'rank_v1_length_first': report.get('rank_v1_length_first'),
            'rank_v2': report.get('rank_v2'),
            'rank_delta_v2_minus_v1': report.get('rank_delta_v2_minus_v1'),
            'continuity_tier': report.get('continuity_tier'),
            'q_floor': report.get('q_floor'),
            'q_sum': report.get('q_sum'),
            'selected_for_marking': bool(report.get('selected_for_marking', False)),
            'marked': bool(report.get('accepted', False)),
            'skipped_reason': report.get('skipped_reason') or 'not_found',
            'candidate_score_tuple': report.get('candidate_score_tuple_v2'),
            'candidate_score_tuple_v2': report.get('candidate_score_tuple_v2'),
        })
    return human_reports


def _human_gap_labels_for_path(path):
    canonical = _canonical_two_edge_path(path)
    return [
        label for label, _, _, human_path in DIAGNOSTIC_HUMAN_GAP_REGRESSION_PATHS
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


def _two_edge_endpoint_bridge_sort_key_v1(report):
    min_alignment = report['min_endpoint_tangent_alignment']
    return (
        report['total_path_length'],
        report['endpoint_distance'],
        -min_alignment,
        -report['path_straightness'],
        tuple(report['path_vertex_ids']),
    )


def _two_edge_endpoint_bridge_sort_key_v2(report):
    return (
        _endpoint_bridge_continuity_tier(report),
        -min(report['min_endpoint_tangent_alignment'], report['path_straightness']),
        -(report['min_endpoint_tangent_alignment'] + report['path_straightness']),
        report['total_path_length'],
        report['endpoint_distance'],
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
    vertices = getattr(mesh, 'vertices', None)
    if vertices is None:
        return None
    positions = [
        position for position in (_vertex_position(mesh, index) for index in range(len(vertices)))
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


DIAGNOSTIC_HUMAN_CASE_2557_2558 = (2557, 2558)


DIAGNOSTIC_HUMAN_GAP_REGRESSION_PATHS = (
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


DIAGNOSTIC_RESIDUAL_GAP_PHASE2E_PATHS = (
    ('2', '2', 'main', (234, 319, 318, 214)),
    ('3a', '3', 'a', (3098, 3185, 3192)),
    ('3b', '3', 'b', (3098, 3097, 3192)),
    ('5', '5', 'main', (5477, 5520, 5483)),
    ('6a', '6', 'a', (5562, 5464, 5553)),
    ('6b', '6', 'b', (5562, 5463, 5553)),
    ('8a', '8', 'a', (5149, 3003, 3005)),
    ('8b', '8', 'b', (5149, 5103, 3005)),
    ('9', '9', 'main', (3006, 3008, 3039)),
    ('13a', '13', 'a', (670, 669, 666)),
    ('13b', '13', 'b', (670, 671, 666)),
    ('14a', '14', 'a', (3, 2438, 2205)),
    ('14b', '14', 'b', (3, 700, 2205)),
    ('15a', '15', 'a', (2391, 1723)),
    ('15b', '15', 'b', (1734, 1723, 1722)),
)


DIAGNOSTIC_OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS = (
    ('target_a_2045_2541_4884', (2045, 2541, 4884)),
    ('target_b_2540_2541_2544', (2540, 2541, 2544)),
)


def _diagnostic_old_endpoint_bridge_target_keys():
    return {
        _canonical_two_edge_path(path)
        for _, path in DIAGNOSTIC_OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS
    }


ENDPOINT_BRIDGE_SCORE_TUPLE_DEFINITION_V1_LENGTH_FIRST = [
    'total_path_length',
    'endpoint_distance',
    '-min_endpoint_tangent_alignment',
    '-path_straightness',
    'path_vertex_ids',
]

ENDPOINT_BRIDGE_SCORE_TUPLE_DEFINITION_V2 = [
    'continuity_tier',
    '-q_floor',
    '-q_sum',
    'total_path_length',
    'endpoint_distance',
    'path_vertex_ids',
]


def classify_human_gap_regressions(
    mesh,
    paths=DIAGNOSTIC_HUMAN_GAP_REGRESSION_PATHS,
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


def classify_residual_gap_phase2e(
    mesh,
    paths=DIAGNOSTIC_RESIDUAL_GAP_PHASE2E_PATHS,
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
        reports.append(_classify_residual_gap_phase2e_path(
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

    _annotate_residual_phase2e_special_reports(reports)
    summary = _residual_phase2e_summary(reports)
    if tuple(bool(edge.use_seam) for edge in mesh.edges) != seam_flags_before:
        raise RuntimeError('Phase 2E residual classifier modified seam flags.')
    return {
        'summary': summary,
        'paths': reports,
        'read_only': True,
    }


def _classify_residual_gap_phase2e_path(
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
    endpoint_report = None
    if len(path) == 3:
        endpoint_report = endpoint_bridge_report_by_path.get(_canonical_two_edge_path(path))
    class_phase2e = _residual_phase2e_candidate_class(
        path=path,
        all_edges_exist=all_edges_exist,
        all_marked=all_marked,
        relation=relation,
        seam_degree=seam_degree,
        intermediate_vertices=intermediate_vertices,
        endpoint_report=endpoint_report,
        phase_2b1_allowed=phase_2b1_allowed,
        phase_2b1_reason=phase_2b1_reason,
    )
    rank_v2 = None if endpoint_report is None else endpoint_report.get('rank_v2')
    threshold = None if endpoint_report is None else endpoint_report.get('selected_rank_threshold')
    selected = bool(endpoint_report and endpoint_report.get('selected_for_marking', False))
    duplicate_suppressed = bool(endpoint_report and endpoint_report.get('duplicate_endpoint_pair_suppressed', False))
    skipped_reason = None if endpoint_report is None else endpoint_report.get('skipped_reason')
    rank_delta = None
    if rank_v2 is not None:
        rank_delta = rank_v2 - 9

    report = {
        'label': label,
        'preferred_group_id': group_id,
        'alternative_id': alternative_id,
        'path_vertex_ids': [int(vertex) for vertex in path],
        'path_length_edges': len(edge_keys),
        'all_edges_exist_in_blender': bool(all_edges_exist),
        'edge_keys': [[int(a), int(b)] for a, b in edge_keys],
        'blender_edge_indices': [None if record is None else int(record[0]) for record in edge_records],
        'missing_edge_keys': missing_edge_keys,
        'edge_seam_flags_after_all_repairs': edge_flags,
        'already_all_marked': bool(all_marked),
        'already_partially_marked': bool(partially_marked),
        'marked_by_prediction_if_traceable': bool(any(edge_key in predicted_key_set for edge_key in edge_keys)),
        'marked_by_phase_2a1_if_traceable': bool(any(edge_key in local_marked for edge_key in edge_keys)),
        'marked_by_phase_2b_same_component_if_traceable': bool(
            any(edge_key in two_edge_marked for edge_key in edge_keys)
        ),
        'marked_by_phase_2b1_endpoint_bridge_if_traceable': bool(
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
        'would_be_allowed_by_phase_2b1_endpoint_bridge': bool(phase_2b1_allowed),
        'phase_2b1_rejection_reason': phase_2b1_reason,
        'rank_v2_if_available': rank_v2,
        'continuity_tier_if_available': None if endpoint_report is None else endpoint_report.get('continuity_tier'),
        'q_floor_if_available': None if endpoint_report is None else endpoint_report.get('q_floor'),
        'q_sum_if_available': None if endpoint_report is None else endpoint_report.get('q_sum'),
        'selected_for_marking': selected,
        'duplicate_endpoint_pair_suppressed': duplicate_suppressed,
        'skipped_reason': skipped_reason,
        'candidate_class_phase2e': class_phase2e,
        'is_ranking_cap_issue': class_phase2e in (
            'phase_2b1_rank_below_cap',
            'phase_2b1_duplicate_suppressed',
        ),
        'is_new_repair_class_issue': class_phase2e in (
            'three_edge_local_bridge',
            'endpoint_to_skeleton_or_near_junction',
            'same_component_two_edge_local_bridge',
        ),
        'is_visual_or_apply_verification_issue': class_phase2e == 'already_marked_but_human_still_sees_gap',
        'is_not_repairable_without_topology_change': class_phase2e == 'non_original_or_missing_blender_edge',
        'recommended_followup': _residual_phase2e_followup(class_phase2e),
        'would_be_allowed_by_phase_2a1': bool(phase_2a1_allowed),
        'phase_2a1_rejection_reason': phase_2a1_reason,
        'would_be_allowed_by_phase_2b_same_component': bool(phase_2b_allowed),
        'phase_2b_rejection_reason': phase_2b_reason,
        'rank_delta_from_cap': rank_delta,
    }
    if label == '9':
        report.update(_residual_special_3006_report(report, endpoint_report))
    elif label == '8a':
        report.update({
            'is_candidate_for_future_cap_rank_visual_review': bool(
                class_phase2e == 'phase_2b1_rank_below_cap'
            ),
            'is_highest_ranked_unselected_human_candidate': False,
        })
    elif label == '15a':
        report.update({
            'one_endpoint_is_non_seam': not all(report['endpoint_seam_vertex_flags']),
            'why_phase_2a1_does_not_apply': phase_2a1_reason,
        })
    elif label == '15b':
        report.update({
            'same_component_status': relation == 'same_component',
            'degree_pattern': report['vertex_seam_degrees_current'],
            'why_phase_2b1_rejects_it': phase_2b1_reason,
            'same_component_local_closure_candidate': (
                class_phase2e == 'same_component_two_edge_local_bridge'
            ),
        })
    return report


def _residual_phase2e_candidate_class(
    *,
    path,
    all_edges_exist,
    all_marked,
    relation,
    seam_degree,
    intermediate_vertices,
    endpoint_report,
    phase_2b1_allowed,
    phase_2b1_reason,
):
    if not all_edges_exist:
        return 'non_original_or_missing_blender_edge'
    if all_marked:
        return 'already_marked_but_human_still_sees_gap'
    if endpoint_report is not None:
        if endpoint_report.get('duplicate_endpoint_pair_suppressed', False):
            return 'phase_2b1_duplicate_suppressed'
        if endpoint_report.get('rejection_reason') == 'tangent_alignment_failed':
            return 'phase_2b1_tangent_failed'
        if (
            phase_2b1_allowed
            and not endpoint_report.get('selected_for_marking', False)
            and endpoint_report.get('skipped_reason') == 'over_cap_ranked_below_threshold'
        ):
            return 'phase_2b1_rank_below_cap'
    if len(path) == 4:
        return 'three_edge_local_bridge'
    if len(path) == 3 and relation == 'same_component':
        return 'same_component_two_edge_local_bridge'
    if len(path) == 2 and any(seam_degree.get(vertex, 0) > 0 for vertex in path):
        return 'endpoint_to_skeleton_or_near_junction'
    if any(seam_degree.get(vertex, 0) > 0 for vertex in intermediate_vertices):
        return 'endpoint_to_skeleton_or_near_junction'
    if phase_2b1_reason == 'tangent_alignment_failed':
        return 'phase_2b1_tangent_failed'
    return 'unsupported_or_unknown'


def _residual_phase2e_followup(candidate_class):
    return {
        'already_marked_but_human_still_sees_gap': 'inspect_already_marked_visual_mismatch',
        'phase_2b1_rank_below_cap': 'review_phase_2b1_rank_9_to_16',
        'phase_2b1_duplicate_suppressed': 'review_phase_2b1_rank_9_to_16',
        'phase_2b1_tangent_failed': 'review_phase_2b1_rank_9_to_16',
        'three_edge_local_bridge': 'add_three_edge_classifier_before_repair',
        'endpoint_to_skeleton_or_near_junction': 'investigate_endpoint_to_skeleton',
        'same_component_two_edge_local_bridge': 'investigate_same_component_local_closure',
        'non_original_or_missing_blender_edge': 'investigate_non_original_edges',
        'unsupported_or_unknown': 'no_dominant_next_action',
    }.get(candidate_class, 'no_dominant_next_action')


def _residual_special_3006_report(report, endpoint_report):
    selected_before = bool(
        endpoint_report
        and endpoint_report.get('rank_v1_length_first') is not None
        and endpoint_report.get('rank_v1_length_first') <= 8
    )
    if report['already_all_marked']:
        reason = 'already_marked_after_phase_2d2'
    elif selected_before and report.get('path_straightness') is not None and report['path_straightness'] < 0.5:
        reason = 'v2_continuity_ranking_demoted_weak_straightness'
    elif selected_before:
        reason = 'v2_continuity_ranking_changed_order'
    else:
        reason = 'not_selected_by_v2_ranking'
    return {
        'is_marked_after_phase_2d2': bool(report['already_all_marked']),
        'why_selected_before_phase_2d2_but_not_now': reason,
        'current_status_is_ranking_outcome': bool(
            report['candidate_class_phase2e'] in (
                'phase_2b1_rank_below_cap',
                'phase_2b1_duplicate_suppressed',
                'phase_2b1_tangent_failed',
            )
        ),
        'current_status_is_apply_or_display_mismatch': bool(report['already_all_marked']),
    }


def _annotate_residual_phase2e_special_reports(reports):
    unselected = [
        report for report in reports
        if (
            report.get('would_be_allowed_by_phase_2b1_endpoint_bridge')
            and not report.get('selected_for_marking')
            and report.get('rank_v2_if_available') is not None
        )
    ]
    highest_rank = min((report['rank_v2_if_available'] for report in unselected), default=None)
    for report in reports:
        if report['label'] == '8a':
            report['is_highest_ranked_unselected_human_candidate'] = bool(
                highest_rank is not None
                and report.get('rank_v2_if_available') == highest_rank
                and not report.get('selected_for_marking')
            )
            report['is_candidate_for_future_cap_rank_visual_review'] = bool(
                report.get('rank_delta_from_cap') is not None
                and report['rank_delta_from_cap'] >= 1
                and report.get('candidate_class_phase2e') == 'phase_2b1_rank_below_cap'
            )


def _residual_phase2e_summary(reports):
    count_by_class = _count_by_key(reports, 'candidate_class_phase2e')
    total = len(reports)
    summary = {
        'residual_paths_total': total,
        'residual_paths_editable': sum(1 for report in reports if report['all_edges_exist_in_blender']),
        'residual_paths_missing_blender_edges': sum(
            1 for report in reports if not report['all_edges_exist_in_blender']
        ),
        'residual_paths_already_all_marked': count_by_class.get(
            'already_marked_but_human_still_sees_gap', 0
        ),
        'residual_paths_phase_2b1_rank_below_cap': count_by_class.get('phase_2b1_rank_below_cap', 0),
        'residual_paths_phase_2b1_duplicate_suppressed': count_by_class.get(
            'phase_2b1_duplicate_suppressed', 0
        ),
        'residual_paths_phase_2b1_tangent_failed': count_by_class.get('phase_2b1_tangent_failed', 0),
        'residual_paths_three_edge_local_bridge': count_by_class.get('three_edge_local_bridge', 0),
        'residual_paths_endpoint_to_skeleton_or_near_junction': count_by_class.get(
            'endpoint_to_skeleton_or_near_junction', 0
        ),
        'residual_paths_same_component_two_edge_local_bridge': count_by_class.get(
            'same_component_two_edge_local_bridge', 0
        ),
        'residual_paths_non_original_or_missing_blender_edge': count_by_class.get(
            'non_original_or_missing_blender_edge', 0
        ),
        'residual_paths_unsupported_or_unknown': count_by_class.get('unsupported_or_unknown', 0),
        'count_by_candidate_class_phase2e': count_by_class,
    }
    summary['recommended_next_action'] = _recommended_residual_phase2e_next_action(summary)
    return summary


def _recommended_residual_phase2e_next_action(summary):
    candidates = [
        ('inspect_already_marked_visual_mismatch', summary['residual_paths_already_all_marked']),
        ('review_phase_2b1_rank_9_to_16', (
            summary['residual_paths_phase_2b1_rank_below_cap']
            + summary['residual_paths_phase_2b1_duplicate_suppressed']
            + summary['residual_paths_phase_2b1_tangent_failed']
        )),
        ('add_three_edge_classifier_before_repair', summary['residual_paths_three_edge_local_bridge']),
        ('investigate_endpoint_to_skeleton', summary['residual_paths_endpoint_to_skeleton_or_near_junction']),
        ('investigate_same_component_local_closure', summary['residual_paths_same_component_two_edge_local_bridge']),
        ('investigate_non_original_edges', summary['residual_paths_non_original_or_missing_blender_edge']),
    ]
    best_action, best_count = max(candidates, key=lambda item: item[1])
    if best_count <= 0:
        return 'no_dominant_next_action'
    tied = [action for action, count in candidates if count == best_count]
    if len(tied) > 1:
        return 'no_dominant_next_action'
    return best_action


PHASE2H_CANDIDATE_DETAIL_CAPS = {
    'residual_matched': None,
    'current_selected_repair': 50,
    'high_priority_unselected': 50,
    'medium_priority_unselected': 25,
    'low_priority_or_unsafe_per_class': 10,
    'three_edge_local_bridge': 50,
}


def _phase2h_collect_candidate_reports(
    mesh,
    predicted_keys=None,
    local_repair_reports=(),
    two_edge_reports=(),
    endpoint_bridge_reports=(),
    residual_payload=None,
):
    edge_items, edge_by_key, adjacency = _mesh_edge_lookup(mesh)
    seam_degree, seam_adjacency = _seam_topology_from_mesh_edges(edge_items)
    component_id_of = _seam_component_ids(seam_adjacency)
    bbox_diagonal = _mesh_bbox_diagonal(mesh)
    predicted_key_set = set(predicted_keys or ())
    local_marked = _marked_edge_key_set(local_repair_reports)
    two_edge_marked = _marked_edge_key_set(two_edge_reports)
    endpoint_bridge_marked = _marked_edge_key_set(endpoint_bridge_reports)
    two_edge_report_by_path = _report_by_path(two_edge_reports)
    endpoint_bridge_report_by_path = _report_by_path(endpoint_bridge_reports)
    residual_by_path = _phase2h_residual_reports_by_path(residual_payload)
    discovered = {}

    def add_path(path, source):
        key = _phase2h_path_key(path)
        if key not in discovered:
            discovered[key] = {
                'path': key,
                'sources': set(),
            }
        discovered[key]['sources'].add(source)

    for _, key, edge in edge_items:
        if edge.use_seam:
            continue
        if seam_degree.get(key[0], 0) > 0 or seam_degree.get(key[1], 0) > 0:
            add_path(key, 'length_1_unmarked')

    for middle in sorted(adjacency):
        neighbors = sorted(adjacency[middle])
        for left_index in range(len(neighbors)):
            for right_index in range(left_index + 1, len(neighbors)):
                path = _canonical_two_edge_path((neighbors[left_index], middle, neighbors[right_index]))
                edge_keys = _two_edge_path_edge_keys(path)
                if all(edge_by_key[edge_key][1].use_seam is False for edge_key in edge_keys):
                    add_path(path, 'length_2_unmarked')

    seam_vertices = sorted(vertex for vertex, degree in seam_degree.items() if degree > 0)
    for u in seam_vertices:
        for a in sorted(adjacency.get(u, ())):
            edge_ua = (min(u, a), max(u, a))
            if edge_by_key[edge_ua][1].use_seam:
                continue
            for b in sorted(adjacency.get(a, ())):
                if b in (u, a):
                    continue
                edge_ab = (min(a, b), max(a, b))
                if edge_by_key[edge_ab][1].use_seam:
                    continue
                for v in sorted(adjacency.get(b, ())):
                    if v in (u, a, b) or seam_degree.get(v, 0) <= 0:
                        continue
                    edge_bv = (min(b, v), max(b, v))
                    if edge_by_key[edge_bv][1].use_seam:
                        continue
                    add_path(_phase2h_canonical_path((u, a, b, v)), 'length_3_unmarked')

    for report in local_repair_reports:
        if isinstance(report, dict) and report.get('accepted') and report.get('vertex_ids_0based'):
            add_path(tuple(report['vertex_ids_0based']), 'current_selected_repair')
    for report in tuple(two_edge_reports) + tuple(endpoint_bridge_reports):
        if isinstance(report, dict) and report.get('accepted') and report.get('path_vertex_ids'):
            add_path(tuple(report['path_vertex_ids']), 'current_selected_repair')

    for residual in residual_by_path.values():
        path = tuple(residual.get('path_vertex_ids', ()))
        if path:
            add_path(path, 'residual_path')

    reports = []
    for index, item in enumerate(sorted(discovered.values(), key=lambda entry: entry['path']), start=1):
        reports.append(_phase2h_candidate_report(
            mesh=mesh,
            candidate_index=index,
            path=item['path'],
            sources=item['sources'],
            edge_by_key=edge_by_key,
            seam_degree=seam_degree,
            seam_adjacency=seam_adjacency,
            component_id_of=component_id_of,
            bbox_diagonal=bbox_diagonal,
            predicted_key_set=predicted_key_set,
            local_marked=local_marked,
            two_edge_marked=two_edge_marked,
            endpoint_bridge_marked=endpoint_bridge_marked,
            two_edge_report_by_path=two_edge_report_by_path,
            endpoint_bridge_report_by_path=endpoint_bridge_report_by_path,
            residual_by_path=residual_by_path,
        ))

    return reports, residual_by_path


def collect_general_residual_candidates_phase2h(
    mesh,
    predicted_keys=None,
    local_repair_reports=(),
    two_edge_reports=(),
    endpoint_bridge_reports=(),
    residual_payload=None,
):
    seam_flags_before = tuple(bool(edge.use_seam) for edge in mesh.edges)
    reports, residual_by_path = _phase2h_collect_candidate_reports(
        mesh,
        predicted_keys=predicted_keys,
        local_repair_reports=local_repair_reports,
        two_edge_reports=two_edge_reports,
        endpoint_bridge_reports=endpoint_bridge_reports,
        residual_payload=residual_payload,
    )
    stored_reports, truncation = _phase2h_apply_detail_caps(reports)
    residual_mapping = _phase2h_residual_mapping(residual_by_path, reports, stored_reports)
    summary = _phase2h_summary(reports, stored_reports, truncation, residual_mapping)
    if tuple(bool(edge.use_seam) for edge in mesh.edges) != seam_flags_before:
        raise RuntimeError('Phase 2H candidate collector modified seam flags.')
    return {
        'summary': summary,
        'candidate_detail_caps': dict(PHASE2H_CANDIDATE_DETAIL_CAPS),
        'candidates': stored_reports,
        'human_residual_mapping': residual_mapping,
        'r_bridge_value_if_available': None,
        'r_bridge_source_if_available': None,
        'bridge_radius_related_fields_if_available': {},
        'read_only': True,
    }


def _phase2h_candidate_report(
    *,
    mesh,
    candidate_index,
    path,
    sources,
    edge_by_key,
    seam_degree,
    seam_adjacency,
    component_id_of,
    bbox_diagonal,
    predicted_key_set,
    local_marked,
    two_edge_marked,
    endpoint_bridge_marked,
    two_edge_report_by_path,
    endpoint_bridge_report_by_path,
    residual_by_path,
):
    edge_keys = _path_edge_keys(path)
    edge_records = [edge_by_key.get(edge_key) for edge_key in edge_keys]
    edge_flags = [None if record is None else bool(record[1].use_seam) for record in edge_records]
    all_edges_exist = all(record is not None for record in edge_records)
    endpoint_vertices = (path[0], path[-1])
    intermediate_vertices = path[1:-1]
    component_ids = [component_id_of.get(vertex) for vertex in path]
    relation = _phase2h_component_relation(endpoint_vertices, component_id_of, seam_degree)
    seam_distance = None
    if relation == 'same_component':
        seam_distance = _shortest_seam_path_length(seam_adjacency, path[0], path[-1])
    total_path_length, endpoint_distance, straightness = _path_geometry(mesh, path)
    tangent_alignments, tangent_flags = _phase2h_tangent_alignments(mesh, path, seam_adjacency)
    min_alignment = None
    if tangent_alignments is not None and all(value is not None for value in tangent_alignments):
        min_alignment = min(tangent_alignments)
    q_floor, q_sum, continuity_tier = _phase2h_quality(min_alignment, straightness)
    endpoint_report = None
    phase_2b1_allowed = False
    phase_2b1_reason = 'path_length_not_supported'
    rank_v2 = None
    if len(path) == 3:
        canonical = _canonical_two_edge_path(path)
        endpoint_report = endpoint_bridge_report_by_path.get(canonical)
        phase_2b1_allowed, phase_2b1_reason = _classification_phase_2b1(
            canonical,
            endpoint_bridge_report_by_path,
            edge_records,
            seam_degree,
            component_id_of,
        )
        if endpoint_report is not None:
            rank_v2 = endpoint_report.get('rank_v2')
    selected_by_current = bool(
        'current_selected_repair' in sources
        or (endpoint_report is not None and endpoint_report.get('accepted', False))
    )
    marked_trace = bool(
        any(edge_key in local_marked for edge_key in edge_keys)
        or any(edge_key in two_edge_marked for edge_key in edge_keys)
        or any(edge_key in endpoint_bridge_marked for edge_key in edge_keys)
    )
    residual = residual_by_path.get(_phase2h_path_key(path))
    candidate_class = _phase2h_candidate_class(
        path=path,
        all_edges_exist=all_edges_exist,
        selected_by_current=selected_by_current,
        relation=relation,
        seam_degree=seam_degree,
        endpoint_report=endpoint_report,
        phase_2b1_reason=phase_2b1_reason,
    )
    priority = _phase2h_priority(candidate_class, residual, continuity_tier, q_floor)
    return {
        'candidate_id': f"phase2h_{candidate_index:05d}",
        'path_vertex_ids': [int(vertex) for vertex in path],
        'path_length_edges': len(edge_keys),
        'path_edge_keys': [[int(a), int(b)] for a, b in edge_keys],
        'blender_edge_indices': [None if record is None else int(record[0]) for record in edge_records],
        'edge_seam_flags_after_all_repairs': edge_flags,
        'endpoint_seam_vertex_flags': [bool(seam_degree.get(vertex, 0) > 0) for vertex in endpoint_vertices],
        'intermediate_seam_vertex_flags': [
            bool(seam_degree.get(vertex, 0) > 0) for vertex in intermediate_vertices
        ],
        'vertex_seam_degrees': [int(seam_degree.get(vertex, 0)) for vertex in path],
        'degree_pattern': [int(seam_degree.get(vertex, 0)) for vertex in path],
        'component_ids': component_ids,
        'component_relation': relation,
        'would_create_loop': relation == 'same_component',
        'existing_seam_distance_between_endpoints_if_available': seam_distance,
        'duplicate_endpoint_pair_key': sorted([int(path[0]), int(path[-1])]),
        'duplicate_group_rank_if_available': None if endpoint_report is None else endpoint_report.get('rank_v2'),
        'total_path_length': total_path_length,
        'endpoint_distance': endpoint_distance,
        'normalized_total_path_length_if_mesh_scale_available': _safe_ratio(total_path_length, bbox_diagonal),
        'normalized_endpoint_distance_if_mesh_scale_available': _safe_ratio(endpoint_distance, bbox_diagonal),
        'path_straightness': straightness,
        'endpoint_tangent_alignments': tangent_alignments,
        'min_endpoint_tangent_alignment': min_alignment,
        'tangent_available_flags': tangent_flags,
        'matches_phase_2a1_one_edge_missing_continuity': bool(
            len(path) == 2 and candidate_class == 'one_edge_missing_continuity'
        ),
        'matches_phase_2b1_inter_component_two_edge_endpoint_bridge': bool(
            len(path) == 3 and candidate_class == 'two_edge_inter_component_endpoint_bridge'
        ),
        'would_be_allowed_by_phase_2b1_current_predicate': bool(phase_2b1_allowed),
        'phase_2b1_rejection_reason_if_any': phase_2b1_reason,
        'rank_v2_if_available': rank_v2,
        'selected_by_current_pipeline': selected_by_current,
        'marked_by_current_pipeline_if_traceable': marked_trace,
        'marked_by_prediction_if_traceable': bool(any(edge_key in predicted_key_set for edge_key in edge_keys)),
        'candidate_class_phase2h': candidate_class,
        'continuity_tier_general': continuity_tier,
        'q_floor_general': q_floor,
        'q_sum_general': q_sum,
        'loop_risk': _phase2h_loop_risk(relation, seam_distance),
        'tangent_risk': _phase2h_tangent_risk(tangent_flags, min_alignment),
        'length_risk': _phase2h_length_risk(total_path_length, bbox_diagonal),
        'topology_risk': _phase2h_topology_risk(candidate_class),
        'candidate_priority': priority,
        'would_require_new_repair_class': candidate_class in {
            'one_edge_endpoint_to_skeleton',
            'two_edge_same_component_local_closure',
            'two_edge_endpoint_to_skeleton_or_near_junction',
            'three_edge_local_bridge',
        },
        'would_require_parameter_or_cap_change': candidate_class in {
            'two_edge_duplicate_alternative',
            'two_edge_tangent_failed_endpoint_bridge',
        },
        'would_require_topology_remapping': candidate_class == 'non_original_or_missing_blender_edge',
        'residual_match_labels': [] if residual is None else [residual.get('label')],
        'source_tags': sorted(sources),
    }


def _phase2h_candidate_class(*, path, all_edges_exist, selected_by_current, relation, seam_degree, endpoint_report, phase_2b1_reason):
    if not all_edges_exist:
        return 'non_original_or_missing_blender_edge'
    if selected_by_current:
        return 'current_selected_repair'
    if len(path) == 1:
        return 'unsupported_or_unknown'
    if len(path) == 2:
        endpoint_seams = [seam_degree.get(path[0], 0) > 0, seam_degree.get(path[-1], 0) > 0]
        if all(endpoint_seams):
            return 'one_edge_missing_continuity'
        if any(endpoint_seams):
            return 'one_edge_endpoint_to_skeleton'
        return 'unsupported_or_unknown'
    if len(path) == 3:
        if endpoint_report is not None and endpoint_report.get('duplicate_endpoint_pair_suppressed', False):
            return 'two_edge_duplicate_alternative'
        if phase_2b1_reason == 'tangent_alignment_failed':
            return 'two_edge_tangent_failed_endpoint_bridge'
        middle_degree = seam_degree.get(path[1], 0)
        endpoint_degrees = (seam_degree.get(path[0], 0), seam_degree.get(path[-1], 0))
        if endpoint_degrees == (1, 1) and middle_degree == 0 and relation == 'different_components':
            return 'two_edge_inter_component_endpoint_bridge'
        if relation == 'same_component':
            return 'two_edge_same_component_local_closure'
        if middle_degree > 0 or any(degree > 0 for degree in endpoint_degrees):
            return 'two_edge_endpoint_to_skeleton_or_near_junction'
        return 'unsupported_or_unknown'
    if len(path) == 4:
        return 'three_edge_local_bridge'
    return 'unsupported_or_unknown'


def _phase2h_apply_detail_caps(reports):
    reports = sorted(reports, key=_phase2h_sort_key)
    stored = []
    stored_ids = set()
    per_class_stored = {}
    for report in reports:
        if report['residual_match_labels']:
            stored.append(report)
            stored_ids.add(report['candidate_id'])
            per_class_stored[report['candidate_class_phase2h']] = (
                per_class_stored.get(report['candidate_class_phase2h'], 0) + 1
            )
    for report in reports:
        if report['candidate_id'] in stored_ids:
            continue
        class_name = report['candidate_class_phase2h']
        cap = _phase2h_cap_for_report(report)
        current = per_class_stored.get(class_name, 0)
        if cap is not None and current >= cap:
            continue
        stored.append(report)
        stored_ids.add(report['candidate_id'])
        per_class_stored[class_name] = current + 1
    stored.sort(key=_phase2h_sort_key)
    discovered_counts = _count_by_key(reports, 'candidate_class_phase2h')
    stored_counts = _count_by_key(stored, 'candidate_class_phase2h')
    truncation = {
        class_name: max(0, count - stored_counts.get(class_name, 0))
        for class_name, count in discovered_counts.items()
    }
    return stored, {
        'per_class_discovered_counts': discovered_counts,
        'per_class_stored_counts': stored_counts,
        'per_class_truncation_counts': truncation,
    }


def _phase2h_cap_for_report(report):
    if report['candidate_class_phase2h'] == 'current_selected_repair':
        return PHASE2H_CANDIDATE_DETAIL_CAPS['current_selected_repair']
    if report['candidate_class_phase2h'] == 'three_edge_local_bridge':
        return PHASE2H_CANDIDATE_DETAIL_CAPS['three_edge_local_bridge']
    if report['candidate_priority'] == 'high':
        return PHASE2H_CANDIDATE_DETAIL_CAPS['high_priority_unselected']
    if report['candidate_priority'] == 'medium':
        return PHASE2H_CANDIDATE_DETAIL_CAPS['medium_priority_unselected']
    return PHASE2H_CANDIDATE_DETAIL_CAPS['low_priority_or_unsafe_per_class']


def _phase2h_residual_mapping(residual_by_path, reports, stored_reports):
    by_path = {}
    for report in reports:
        by_path.setdefault(_phase2h_path_key(report['path_vertex_ids']), []).append(report)
    stored_ids = {report['candidate_id'] for report in stored_reports}
    mappings = []
    for path, residual in sorted(residual_by_path.items()):
        matches = sorted(by_path.get(path, []), key=_phase2h_sort_key)
        best = matches[0] if matches else None
        class_name = None if best is None else best['candidate_class_phase2h']
        mappings.append({
            'residual_label': residual.get('label'),
            'path_vertex_ids': list(path),
            'matched_candidate_ids': [report['candidate_id'] for report in matches],
            'best_matching_candidate_id': None if best is None else best['candidate_id'],
            'current_status': residual.get('candidate_class_phase2e'),
            'current_rejection_or_skip_reason': (
                residual.get('phase_2b1_rejection_reason')
                or residual.get('skipped_reason')
                or residual.get('phase_2b_rejection_reason')
                or residual.get('phase_2a1_rejection_reason')
            ),
            'generalized_candidate_class': class_name,
            'whether_similar_candidates_exist_beyond_the_listed_residual': bool(
                class_name and sum(1 for report in reports if report['candidate_class_phase2h'] == class_name) > len(matches)
            ),
            'count_of_similar_candidates': 0 if class_name is None else sum(
                1 for report in reports if report['candidate_class_phase2h'] == class_name
            ),
            'recommended_followup': residual.get('recommended_followup') or _phase2h_followup_for_class(class_name),
            'candidate_generation_cap_truncated': bool(
                best is not None and best['candidate_id'] not in stored_ids
            ),
            'unmatched_reason': _phase2h_unmatched_reason(residual, matches),
        })
    return mappings


def _phase2h_summary(reports, stored_reports, truncation, residual_mapping):
    candidates_by_length = _count_by_key(reports, 'path_length_edges')
    candidates_by_class = _count_by_key(reports, 'candidate_class_phase2h')
    candidates_by_relation = _count_by_key(reports, 'component_relation')
    candidates_by_priority = _count_by_key(reports, 'candidate_priority')
    residual_coverage = {}
    for mapping in residual_mapping:
        class_name = mapping.get('generalized_candidate_class') or 'unmatched'
        residual_coverage[class_name] = residual_coverage.get(class_name, 0) + 1
    total_truncated = sum(truncation['per_class_truncation_counts'].values())
    summary = {
        'total_candidates_discovered_before_truncation': len(reports),
        'total_candidates_stored_after_truncation': len(stored_reports),
        'total_candidates_truncated': total_truncated,
        'candidates_by_path_length': candidates_by_length,
        'candidates_by_class': candidates_by_class,
        'candidates_by_component_relation': candidates_by_relation,
        'candidates_by_priority': candidates_by_priority,
        'candidates_matching_existing_repairs': candidates_by_class.get('current_selected_repair', 0),
        'candidates_requiring_new_repair_class': sum(
            1 for report in reports if report['would_require_new_repair_class']
        ),
        'candidates_requiring_cap_or_ranking_change': sum(
            1 for report in reports if report['would_require_parameter_or_cap_change']
        ),
        'candidates_requiring_topology_remapping': sum(
            1 for report in reports if report['would_require_topology_remapping']
        ),
        'residual_paths_total': len(residual_mapping),
        'residual_paths_matched': sum(1 for mapping in residual_mapping if mapping['matched_candidate_ids']),
        'residual_paths_unmatched': sum(1 for mapping in residual_mapping if not mapping['matched_candidate_ids']),
        'residual_coverage_by_class': residual_coverage,
        'per_class_discovered_counts': truncation['per_class_discovered_counts'],
        'per_class_stored_counts': truncation['per_class_stored_counts'],
        'per_class_truncation_counts': truncation['per_class_truncation_counts'],
        'candidate_detail_caps': dict(PHASE2H_CANDIDATE_DETAIL_CAPS),
    }
    summary['recommended_next_action'] = _phase2h_recommendation(summary)
    return summary


def _phase2h_recommendation(summary):
    residual_total = max(1, summary['residual_paths_total'])
    coverage = summary['residual_coverage_by_class']
    if coverage.get('non_original_or_missing_blender_edge', 0) >= residual_total * 0.4:
        return 'investigate_non_original_edges'
    dominant = max(coverage.items(), key=lambda item: item[1], default=(None, 0))
    if dominant[1] >= residual_total * 0.4:
        class_name = dominant[0]
        if class_name == 'three_edge_local_bridge':
            return 'consider_three_edge_classifier'
        if class_name in ('one_edge_endpoint_to_skeleton', 'two_edge_endpoint_to_skeleton_or_near_junction'):
            return 'consider_endpoint_to_skeleton_classifier'
        if class_name == 'two_edge_same_component_local_closure':
            return 'consider_same_component_local_closure_classifier'
        if class_name in ('two_edge_inter_component_endpoint_bridge', 'two_edge_duplicate_alternative', 'two_edge_tangent_failed_endpoint_bridge'):
            return 'review_phase_2b1_cap_or_ranking'
    if len([value for value in coverage.values() if value > 0]) > 2:
        return 'no_single_dominant_next_action'
    return 'review_general_candidate_distribution'


def _phase2h_residual_reports_by_path(residual_payload):
    result = {}
    for report in (residual_payload or {}).get('paths', []):
        path = report.get('path_vertex_ids')
        if isinstance(path, list) and len(path) >= 2:
            result[_phase2h_path_key(path)] = report
    return result


def _phase2h_path_key(path):
    path = tuple(int(vertex) for vertex in path)
    if len(path) == 3:
        return _canonical_two_edge_path(path)
    return _phase2h_canonical_path(path)


def _phase2h_canonical_path(path):
    path = tuple(int(vertex) for vertex in path)
    reverse = tuple(reversed(path))
    return path if path <= reverse else reverse


def _phase2h_component_relation(endpoint_vertices, component_id_of, seam_degree):
    endpoint_flags = [seam_degree.get(vertex, 0) > 0 for vertex in endpoint_vertices]
    if not any(endpoint_flags):
        return 'no_endpoint_seam'
    if not all(endpoint_flags):
        return 'endpoint_not_seam'
    return _component_relation(endpoint_vertices, component_id_of, seam_degree)


def _phase2h_tangent_alignments(mesh, path, seam_adjacency):
    if len(path) == 3:
        return _classification_tangent_alignments(mesh, path, seam_adjacency)
    return None, [False, False]


def _phase2h_quality(min_alignment, straightness):
    if min_alignment is None or straightness is None:
        return None, None, None
    q_floor = min(min_alignment, straightness)
    q_sum = min_alignment + straightness
    report = {
        'min_endpoint_tangent_alignment': min_alignment,
        'path_straightness': straightness,
    }
    return q_floor, q_sum, _endpoint_bridge_continuity_tier(report)


def _phase2h_priority(candidate_class, residual, continuity_tier, q_floor):
    if candidate_class == 'current_selected_repair':
        return 'high'
    if candidate_class == 'non_original_or_missing_blender_edge':
        return 'unsafe'
    if residual is not None:
        return 'high'
    if continuity_tier is not None and continuity_tier <= 1 and (q_floor is None or q_floor >= 0.7):
        return 'high'
    if candidate_class in ('unsupported_or_unknown', 'two_edge_tangent_failed_endpoint_bridge'):
        return 'low'
    if candidate_class == 'three_edge_local_bridge':
        return 'medium'
    return 'medium'


def _phase2h_sort_key(report):
    priority_order = {'high': 0, 'medium': 1, 'low': 2, 'unsafe': 3}
    return (
        priority_order.get(report.get('candidate_priority'), 4),
        report.get('path_length_edges', 99),
        report.get('continuity_tier_general') if report.get('continuity_tier_general') is not None else 99,
        report.get('total_path_length') if report.get('total_path_length') is not None else 10**9,
        tuple(report.get('path_vertex_ids', ())),
    )


def _phase2h_loop_risk(relation, seam_distance):
    if relation != 'same_component':
        return 'none'
    if seam_distance is None:
        return 'unknown'
    if seam_distance <= 3:
        return 'local_loop'
    return 'loop'


def _phase2h_tangent_risk(tangent_flags, min_alignment):
    if not any(tangent_flags):
        return 'unavailable'
    if min_alignment is None:
        return 'unknown'
    if min_alignment < 0:
        return 'failed'
    if min_alignment < 0.5:
        return 'weak'
    return 'low'


def _phase2h_length_risk(total_path_length, bbox_diagonal):
    ratio = _safe_ratio(total_path_length, bbox_diagonal)
    if ratio is None:
        return 'unknown'
    if ratio > 0.03:
        return 'long'
    return 'local'


def _phase2h_topology_risk(candidate_class):
    if candidate_class in ('current_selected_repair', 'one_edge_missing_continuity', 'two_edge_inter_component_endpoint_bridge'):
        return 'accepted_pattern'
    if candidate_class == 'non_original_or_missing_blender_edge':
        return 'not_editable'
    if candidate_class == 'unsupported_or_unknown':
        return 'unknown'
    return 'new_pattern'


def _phase2h_followup_for_class(class_name):
    return {
        'three_edge_local_bridge': 'consider_three_edge_classifier',
        'one_edge_endpoint_to_skeleton': 'consider_endpoint_to_skeleton_classifier',
        'two_edge_endpoint_to_skeleton_or_near_junction': 'consider_endpoint_to_skeleton_classifier',
        'two_edge_same_component_local_closure': 'consider_same_component_local_closure_classifier',
        'two_edge_inter_component_endpoint_bridge': 'review_phase_2b1_cap_or_ranking',
        'two_edge_duplicate_alternative': 'review_phase_2b1_cap_or_ranking',
        'two_edge_tangent_failed_endpoint_bridge': 'review_phase_2b1_cap_or_ranking',
        'non_original_or_missing_blender_edge': 'investigate_non_original_edges',
    }.get(class_name, 'review_general_candidate_distribution')


def _phase2h_unmatched_reason(residual, matches):
    if matches:
        return None
    if not residual.get('all_edges_exist_in_blender', True):
        return 'missing_blender_edge'
    if residual.get('path_length_edges', 0) > 3:
        return 'path_length_out_of_scope'
    if residual.get('already_all_marked', False):
        return 'current_seam_state_already_marked'
    return 'unknown'


PHASE2HR_REPORT_CAPS = {
    'top_rejected_per_class': 10,
    'unsafe_examples_per_class': 5,
}


PHASE2HR_POLICY_NAMES = (
    'conservative_length2_only',
    'conservative_length1_to_3',
    'class_balanced_probe',
)


PHASE2HR_CANONICAL_OPEN_RESIDUAL_PATHS = (
    ('2', (234, 319, 318, 214)),
    ('3a', (3098, 3185, 3192)),
    ('3b', (3098, 3097, 3192)),
    ('5', (5477, 5520, 5483)),
    ('6a', (5562, 5464, 5553)),
    ('6b', (5562, 5463, 5553)),
    ('8b', (5149, 5103, 3005)),
    ('9', (3006, 3008, 3039)),
    ('13a', (670, 669, 666)),
    ('13b', (670, 671, 666)),
    ('14a', (3, 2438, 2205)),
    ('14b', (3, 700, 2205)),
    ('15a', (2391, 1723)),
    ('15b', (1734, 1723, 1722)),
)


PHASE2HR_HIGH_RISK_CLASSES = {
    'one_edge_endpoint_to_skeleton',
    'two_edge_same_component_local_closure',
    'two_edge_tangent_failed_endpoint_bridge',
    'two_edge_endpoint_to_skeleton_or_near_junction',
    'three_edge_local_bridge',
}


def simulate_unified_local_continuity_phase2h_r(
    mesh,
    predicted_keys=None,
    local_repair_reports=(),
    two_edge_reports=(),
    endpoint_bridge_reports=(),
    residual_payload=None,
):
    seam_flags_before = tuple(bool(edge.use_seam) for edge in mesh.edges)
    reports, residual_by_path = _phase2h_collect_candidate_reports(
        mesh,
        predicted_keys=predicted_keys,
        local_repair_reports=local_repair_reports,
        two_edge_reports=two_edge_reports,
        endpoint_bridge_reports=endpoint_bridge_reports,
        residual_payload=residual_payload,
    )
    diagnostic_labels_by_path = _phase2hr_diagnostic_labels_by_path(residual_payload)
    candidates = [
        _phase2hr_candidate_for_simulation(report, diagnostic_labels_by_path)
        for report in reports
    ]
    candidates.sort(key=_phase2hr_sort_key)
    policies = [
        _phase2hr_policy_report(policy_name, candidates, diagnostic_labels_by_path)
        for policy_name in PHASE2HR_POLICY_NAMES
    ]
    residual_coverage = _phase2hr_residual_coverage(
        candidates,
        policies,
        residual_by_path,
        diagnostic_labels_by_path,
    )
    normalized_residual_coverage = _phase2hr_normalized_residual_coverage(
        candidates,
        policies,
        residual_by_path,
        diagnostic_labels_by_path,
    )
    _phase2hr_annotate_policies_with_normalized_metrics(
        policies,
        candidates,
        normalized_residual_coverage,
    )
    normalized_summary = _phase2hr_normalized_summary(policies, normalized_residual_coverage)
    reported_ids = _phase2hr_reported_candidate_ids(policies, candidates)
    per_class_reported_counts = {}
    for candidate in candidates:
        if candidate['candidate_id'] in reported_ids:
            class_name = candidate['candidate_class']
            per_class_reported_counts[class_name] = per_class_reported_counts.get(class_name, 0) + 1
    per_class_total_counts = _count_by_key(candidates, 'candidate_class')
    per_class_truncation_counts = {
        class_name: max(0, count - per_class_reported_counts.get(class_name, 0))
        for class_name, count in per_class_total_counts.items()
    }
    straight_but_tangent_weak = [
        candidate for candidate in candidates
        if candidate['straight_but_tangent_weak']
    ]
    summary = _phase2hr_summary(candidates, policies, residual_coverage)
    seam_flags_after = tuple(bool(edge.use_seam) for edge in mesh.edges)
    if seam_flags_after != seam_flags_before:
        raise RuntimeError('Unified local continuity simulation modified seam flags.')
    return {
        'phase': '2H-R.2',
        'name': 'unified_local_continuity_selector_simulation',
        'read_only': True,
        'seam_flags_unchanged': True,
        'not_applied_to_mesh': True,
        'probabilities_used': False,
        'diagnostic_paths_are_labels_only': True,
        'compact_sidecar': True,
        'total_candidates_considered': len(candidates),
        'total_candidates_reported': len(reported_ids),
        'total_candidates_truncated': max(0, len(candidates) - len(reported_ids)),
        'per_class_reported_counts': per_class_reported_counts,
        'per_class_truncation_counts': per_class_truncation_counts,
        'report_caps': dict(PHASE2HR_REPORT_CAPS),
        'summary': summary,
        'normalized_summary': normalized_summary,
        'normalized_residual_coverage': normalized_residual_coverage,
        'policies': policies,
        'residual_coverage': residual_coverage,
        'straight_but_tangent_weak_candidate_count': len(straight_but_tangent_weak),
        'straight_but_tangent_weak_candidates': straight_but_tangent_weak,
    }


def _phase2hr_candidate_for_simulation(report, diagnostic_labels_by_path):
    path = tuple(report['path_vertex_ids'])
    labels = sorted(set(
        list(report.get('residual_match_labels', ()))
        + diagnostic_labels_by_path.get(_phase2h_path_key(path), [])
    ))
    safety_tier = _phase2hr_safety_tier(report)
    score_tuple = _phase2hr_score_tuple(report, safety_tier)
    edge_flags = list(report['edge_seam_flags_after_all_repairs'])
    all_edges_exist = all(flag is not None for flag in edge_flags)
    already_all_marked = bool(edge_flags and all(flag is True for flag in edge_flags))
    low_straightness = bool(
        report['candidate_class_phase2h'] == 'two_edge_inter_component_endpoint_bridge'
        and report.get('path_straightness') is not None
        and report['path_straightness'] < 0.50
    )
    straight_but_tangent_weak = bool(
        report['path_length_edges'] == 2
        and report['component_relation'] == 'different_components'
        and report.get('path_straightness') is not None
        and report.get('min_endpoint_tangent_alignment') is not None
        and report['path_straightness'] >= 0.85
        and report['min_endpoint_tangent_alignment'] < 0.30
    )
    return {
        'candidate_id': report['candidate_id'],
        'path_vertex_ids': list(report['path_vertex_ids']),
        'path_length_edges': report['path_length_edges'],
        'path_edge_keys': list(report['path_edge_keys']),
        'blender_edge_indices': list(report['blender_edge_indices']),
        'edge_seam_flags_after_all_repairs': list(report['edge_seam_flags_after_all_repairs']),
        'all_edges_exist_in_blender': all_edges_exist,
        'already_all_marked': already_all_marked,
        'endpoint_seam_vertex_flags': list(report['endpoint_seam_vertex_flags']),
        'intermediate_seam_vertex_flags': list(report['intermediate_seam_vertex_flags']),
        'vertex_seam_degrees': list(report['vertex_seam_degrees']),
        'degree_pattern': list(report['degree_pattern']),
        'component_ids': list(report['component_ids']),
        'component_relation': report['component_relation'],
        'would_create_loop': bool(report['would_create_loop']),
        'existing_seam_distance_between_endpoints_if_available': (
            report['existing_seam_distance_between_endpoints_if_available']
        ),
        'total_path_length': report['total_path_length'],
        'endpoint_distance': report['endpoint_distance'],
        'normalized_total_path_length_if_mesh_scale_available': (
            report['normalized_total_path_length_if_mesh_scale_available']
        ),
        'normalized_endpoint_distance_if_mesh_scale_available': (
            report['normalized_endpoint_distance_if_mesh_scale_available']
        ),
        'path_straightness': report['path_straightness'],
        'endpoint_tangent_alignments': report['endpoint_tangent_alignments'],
        'min_endpoint_tangent_alignment': report['min_endpoint_tangent_alignment'],
        'tangent_available_flags': list(report['tangent_available_flags']),
        'duplicate_endpoint_pair_key': list(report['duplicate_endpoint_pair_key']),
        'candidate_class': report['candidate_class_phase2h'],
        'safety_tier': safety_tier,
        'continuity_score_tuple': score_tuple,
        'loop_risk': report['loop_risk'],
        'tangent_risk': report['tangent_risk'],
        'length_risk': report['length_risk'],
        'topology_risk': report['topology_risk'],
        'selected_by_simulation': False,
        'simulation_candidate_for_review': False,
        'not_applied_to_mesh': True,
        'simulated_skip_reason': None,
        'residual_match_labels': labels,
        'selected_low_straightness_two_edge_bridge': low_straightness,
        'low_straightness_warning': low_straightness,
        'visual_review_only_reason': (
            'selected endpoint bridge has low path straightness' if low_straightness else None
        ),
        'straight_but_tangent_weak': straight_but_tangent_weak,
        'possible_tangent_model_false_negative': straight_but_tangent_weak,
        'do_not_select_automatically': straight_but_tangent_weak,
        'would_require_new_repair_class': bool(report['would_require_new_repair_class']),
        'would_require_cap_or_ranking_change': bool(report['would_require_parameter_or_cap_change']),
        'would_require_topology_remapping': bool(report['would_require_topology_remapping']),
        'would_be_allowed_by_phase_2b1_current_predicate': bool(
            report.get('would_be_allowed_by_phase_2b1_current_predicate', False)
        ),
        'rank_v2_if_available': report.get('rank_v2_if_available'),
    }


def _phase2hr_safety_tier(report):
    class_name = report['candidate_class_phase2h']
    if class_name == 'non_original_or_missing_blender_edge':
        return 'unsafe'
    if class_name in ('unsupported_or_unknown', 'two_edge_tangent_failed_endpoint_bridge'):
        return 'low'
    if class_name in ('current_selected_repair', 'one_edge_missing_continuity'):
        return 'high'
    if class_name == 'two_edge_inter_component_endpoint_bridge':
        if report['tangent_risk'] == 'low' and report['length_risk'] == 'local':
            return 'high'
        return 'medium'
    if class_name == 'three_edge_local_bridge' and report['length_risk'] == 'local':
        return 'high'
    return 'medium'


def _phase2hr_score_tuple(report, safety_tier):
    safety_order = {'high': 0, 'medium': 1, 'low': 2, 'unsafe': 3}
    loop_order = {'none': 0, 'local_loop': 1, 'unknown': 2, 'loop': 3}
    component_order = {
        'different_components': 0,
        'same_component': 1,
        'endpoint_not_seam': 2,
        'no_endpoint_seam': 3,
        'unknown': 4,
    }
    min_alignment = report.get('min_endpoint_tangent_alignment')
    straightness = report.get('path_straightness')
    normalized_total = report.get('normalized_total_path_length_if_mesh_scale_available')
    normalized_endpoint = report.get('normalized_endpoint_distance_if_mesh_scale_available')
    tangent_flags = report.get('tangent_available_flags') or []
    return [
        safety_order.get(safety_tier, 4),
        0 if all(tangent_flags) else 1,
        1.0 if min_alignment is None else -float(min_alignment),
        1.0 if straightness is None else -float(straightness),
        999.0 if normalized_total is None else float(normalized_total),
        999.0 if normalized_endpoint is None else float(normalized_endpoint),
        loop_order.get(report.get('loop_risk'), 4),
        component_order.get(report.get('component_relation'), 5),
        1 if report.get('candidate_class_phase2h') == 'two_edge_duplicate_alternative' else 0,
        list(report.get('path_vertex_ids', ())),
    ]


def _phase2hr_sort_key(candidate):
    score = candidate['continuity_score_tuple']
    return tuple(score[:-1]) + (tuple(score[-1]),)


def _phase2hr_policy_report(policy_name, candidates, diagnostic_labels_by_path):
    selected = []
    skipped_by_class = {}
    skipped_reasons = {}
    unsafe_examples = {}
    duplicate_suppressed = 0
    rejected = []
    class_counts = {}
    endpoint_pairs = set()
    for candidate in candidates:
        eligible, reason = _phase2hr_policy_eligibility(policy_name, candidate, class_counts)
        if eligible and _phase2hr_candidate_uses_duplicate_endpoint(candidate, endpoint_pairs):
            eligible = False
            reason = 'duplicate_endpoint_pair_suppressed'
            duplicate_suppressed += 1
        if eligible:
            item = dict(candidate)
            item['selected_by_simulation'] = True
            item['simulation_candidate_for_review'] = True
            item['simulated_skip_reason'] = None
            selected.append(item)
            class_name = item['candidate_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            endpoint_pairs.add(tuple(item['duplicate_endpoint_pair_key']))
            continue
        class_name = candidate['candidate_class']
        skipped_by_class[class_name] = skipped_by_class.get(class_name, 0) + 1
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        item = dict(candidate)
        item['simulated_skip_reason'] = reason
        if reason == 'unsafe_candidate':
            examples = unsafe_examples.setdefault(class_name, [])
            if len(examples) < PHASE2HR_REPORT_CAPS['unsafe_examples_per_class']:
                examples.append(item)
        rejected.append(item)
    selected_by_class = _count_by_key(selected, 'candidate_class')
    residual_paths_selected = sorted({
        label
        for item in selected
        for label in item['residual_match_labels']
    })
    all_labels = sorted({
        label
        for labels in diagnostic_labels_by_path.values()
        for label in labels
    })
    top_rejected = _phase2hr_top_rejected_by_class(rejected)
    return {
        'policy_name': policy_name,
        'selected_total': len(selected),
        'selected_by_class': selected_by_class,
        'selected_candidates': selected,
        'skipped_by_class': skipped_by_class,
        'skipped_reasons': skipped_reasons,
        'residual_paths_selected': residual_paths_selected,
        'residual_paths_still_open': [
            label for label in all_labels if label not in set(residual_paths_selected)
        ],
        'unsafe_candidates_rejected': sum(
            count for reason, count in skipped_reasons.items() if reason == 'unsafe_candidate'
        ),
        'unsafe_candidate_examples_by_class': unsafe_examples,
        'duplicate_candidates_suppressed': duplicate_suppressed,
        'residual_matched_candidates': [
            item if item['candidate_id'] not in {selected_item['candidate_id'] for selected_item in selected}
            else next(selected_item for selected_item in selected if selected_item['candidate_id'] == item['candidate_id'])
            for item in candidates
            if item['residual_match_labels']
        ],
        'top_rejected_candidates_by_class': top_rejected,
        'note': 'read_only_simulation_not_applied',
    }


def _phase2hr_policy_eligibility(policy_name, candidate, selected_by_class):
    if candidate['candidate_class'] == 'non_original_or_missing_blender_edge':
        return False, 'unsafe_candidate'
    if candidate.get('straight_but_tangent_weak', False):
        return False, 'straight_but_tangent_weak_do_not_select_automatically'
    if policy_name == 'conservative_length2_only':
        if candidate['path_length_edges'] != 2:
            return False, 'policy_length_excluded'
        if candidate['candidate_class'] not in ('current_selected_repair', 'two_edge_inter_component_endpoint_bridge'):
            return False, 'policy_class_excluded'
        if sum(selected_by_class.values()) >= 9:
            return False, 'policy_cap_reached'
        return True, None
    if policy_name == 'conservative_length1_to_3':
        if candidate['path_length_edges'] not in (1, 2, 3):
            return False, 'policy_length_excluded'
        if candidate['safety_tier'] != 'high':
            return False, 'safety_tier_not_high'
        caps = {
            'current_selected_repair': 9,
            'one_edge_missing_continuity': 6,
            'two_edge_inter_component_endpoint_bridge': 9,
            'three_edge_local_bridge': 3,
        }
        cap = caps.get(candidate['candidate_class'], 0)
        if selected_by_class.get(candidate['candidate_class'], 0) >= cap:
            return False, 'policy_class_cap_reached'
        return True, None
    if policy_name == 'class_balanced_probe':
        if candidate['candidate_class'] == 'unsupported_or_unknown':
            return False, 'policy_class_excluded'
        if candidate['safety_tier'] == 'unsafe':
            return False, 'unsafe_candidate'
        if selected_by_class.get(candidate['candidate_class'], 0) >= 2:
            return False, 'policy_class_cap_reached'
        return True, None
    return False, 'unknown_policy'


def _phase2hr_candidate_uses_duplicate_endpoint(candidate, endpoint_pairs):
    if candidate['path_length_edges'] < 2:
        return False
    pair = tuple(candidate['duplicate_endpoint_pair_key'])
    return pair in endpoint_pairs or candidate['candidate_class'] == 'two_edge_duplicate_alternative'


def _phase2hr_top_rejected_by_class(rejected):
    result = {}
    counts = {}
    for candidate in sorted(rejected, key=_phase2hr_sort_key):
        class_name = candidate['candidate_class']
        counts[class_name] = counts.get(class_name, 0) + 1
        bucket = result.setdefault(class_name, [])
        if len(bucket) < PHASE2HR_REPORT_CAPS['top_rejected_per_class']:
            bucket.append(candidate)
    for class_name, bucket in result.items():
        for item in bucket:
            item['class_rejected_total_before_truncation'] = counts[class_name]
            item['class_rejected_truncated_count'] = max(0, counts[class_name] - len(bucket))
    return result


def _phase2hr_residual_coverage(candidates, policies, residual_by_path, diagnostic_labels_by_path):
    by_path = {
        _phase2h_path_key(candidate['path_vertex_ids']): candidate
        for candidate in candidates
    }
    by_label = {}
    for path, labels in diagnostic_labels_by_path.items():
        for label in labels:
            by_label[label] = path
    policy_selected = {
        policy['policy_name']: {
            label
            for item in policy['selected_candidates']
            for label in item['residual_match_labels']
        }
        for policy in policies
    }
    coverage = []
    for label, path in sorted(by_label.items()):
        candidate = by_path.get(path)
        residual = residual_by_path.get(path, {})
        selected_by_policy = {
            policy_name: label in labels
            for policy_name, labels in policy_selected.items()
        }
        candidate_class = None if candidate is None else candidate['candidate_class']
        similar_count = 0 if candidate_class is None else sum(
            1 for item in candidates if item['candidate_class'] == candidate_class
        )
        coverage.append({
            'residual_label': label,
            'path_vertex_ids': list(path),
            'current_status': residual.get('candidate_class_phase2e') or (
                None if candidate is None else 'candidate_present_after_current_repairs'
            ),
            'current_class': residual.get('candidate_class_phase2e') or candidate_class,
            'current_rejection_or_skip_reason': (
                residual.get('phase_2b1_rejection_reason')
                or residual.get('skipped_reason')
                or residual.get('phase_2b_rejection_reason')
                or residual.get('phase_2a1_rejection_reason')
            ),
            'unified_candidate_class': candidate_class,
            'selected_by_each_simulation_policy': selected_by_policy,
            'skipped_by_each_simulation_policy': {
                policy['policy_name']: None if selected_by_policy[policy['policy_name']] else (
                    _phase2hr_skip_reason_for_label(policy, label)
                )
                for policy in policies
            },
            'whether_similar_candidates_exist_beyond_the_listed_residual': similar_count > 1,
            'count_of_similar_candidates': similar_count,
        })
    return coverage


def _phase2hr_skip_reason_for_label(policy, label):
    for class_items in policy['top_rejected_candidates_by_class'].values():
        for item in class_items:
            if label in item['residual_match_labels']:
                return item['simulated_skip_reason']
    if label in policy['residual_paths_still_open']:
        return 'not_selected_by_simulation_policy'
    return None


def _phase2hr_diagnostic_labels_by_path(residual_payload):
    result = {}

    def add(label, path):
        result.setdefault(_phase2h_path_key(path), []).append(label)

    for report in (residual_payload or {}).get('paths', []):
        path = report.get('path_vertex_ids')
        label = report.get('label')
        if isinstance(path, list) and len(path) >= 2 and label is not None:
            add(str(label), path)
    for label, _, _, path in DIAGNOSTIC_RESIDUAL_GAP_PHASE2E_PATHS:
        add(f'residual_{label}', path)
    add('solved_human_2557_2558', DIAGNOSTIC_HUMAN_CASE_2557_2558)
    for label, path in DIAGNOSTIC_OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS:
        add(f'solved_{label}', path)
    add('solved_human_8a_5149_3003_3005', (5149, 3003, 3005))
    for path in ((5149, 5103, 3005),):
        add('residual_8b', path)
    for path, labels in result.items():
        result[path] = sorted(set(labels))
    return result


def _phase2hr_reported_candidate_ids(policies, candidates):
    reported = set()
    for policy in policies:
        for item in policy['selected_candidates']:
            reported.add(item['candidate_id'])
        for item in policy['residual_matched_candidates']:
            reported.add(item['candidate_id'])
        for class_items in policy['top_rejected_candidates_by_class'].values():
            for item in class_items:
                reported.add(item['candidate_id'])
        for class_items in policy['unsafe_candidate_examples_by_class'].values():
            for item in class_items:
                reported.add(item['candidate_id'])
    for candidate in candidates:
        if candidate['residual_match_labels']:
            reported.add(candidate['candidate_id'])
    return reported


def _phase2hr_summary(candidates, policies, residual_coverage):
    candidates_by_length = _count_by_key(candidates, 'path_length_edges')
    candidates_by_class = _count_by_key(candidates, 'candidate_class')
    selected_by_policy = {
        policy['policy_name']: policy['selected_total']
        for policy in policies
    }
    residual_by_policy = {
        policy['policy_name']: sum(
            1 for item in residual_coverage
            if item['selected_by_each_simulation_policy'].get(policy['policy_name'], False)
        )
        for policy in policies
    }
    high_risk_classes = sorted(
        class_name for class_name in candidates_by_class
        if class_name in {
            'non_original_or_missing_blender_edge',
            'unsupported_or_unknown',
            'two_edge_tangent_failed_endpoint_bridge',
            'two_edge_same_component_local_closure',
            'three_edge_local_bridge',
        }
    )
    return {
        'candidates_discovered_by_length': candidates_by_length,
        'candidates_discovered_by_class': candidates_by_class,
        'candidates_selected_by_policy': selected_by_policy,
        'residual_coverage_by_policy': residual_by_policy,
        'high_risk_classes': high_risk_classes,
        'recommended_next_action': _phase2hr_recommended_next_action(
            candidates_by_class,
            residual_by_policy,
            len(residual_coverage),
        ),
    }


def _phase2hr_recommended_next_action(candidates_by_class, residual_by_policy, residual_total):
    best_coverage = max(residual_by_policy.values(), default=0)
    if best_coverage > 0:
        return 'inspect_unified_simulation_selected_candidates'
    if candidates_by_class.get('one_edge_endpoint_to_skeleton', 0) > 0:
        return 'design_one_edge_endpoint_to_skeleton_repair'
    if candidates_by_class.get('two_edge_same_component_local_closure', 0) > 0:
        return 'design_two_edge_same_component_closure_repair'
    if candidates_by_class.get('three_edge_local_bridge', 0) > 0:
        return 'design_three_edge_local_bridge_repair'
    if (
        candidates_by_class.get('two_edge_duplicate_alternative', 0)
        or candidates_by_class.get('two_edge_tangent_failed_endpoint_bridge', 0)
    ):
        return 'revise_phase_2b1_ranking_or_cap'
    return 'defer_due_to_unsafe_candidate_distribution'


def _phase2hr_normalized_residual_coverage(candidates, policies, residual_by_path, diagnostic_labels_by_path):
    open_by_label = {
        label: _phase2h_path_key(path)
        for label, path in PHASE2HR_CANONICAL_OPEN_RESIDUAL_PATHS
    }
    open_by_path = {path: label for label, path in open_by_label.items()}
    candidate_by_path = {
        _phase2h_path_key(candidate['path_vertex_ids']): candidate
        for candidate in candidates
    }
    alias_groups = []
    duplicate_label_count = 0
    for label, path in PHASE2HR_CANONICAL_OPEN_RESIDUAL_PATHS:
        key = _phase2h_path_key(path)
        aliases = set(diagnostic_labels_by_path.get(key, ()))
        aliases.add(label)
        aliases.add(f'residual_{label}')
        ordered = _phase2hr_order_aliases(label, aliases)
        duplicate_label_count += max(0, len(ordered) - 1)
        alias_groups.append({
            'canonical_label': label,
            'aliases': ordered,
            'path_vertex_ids': list(key),
        })
    solved_labels = sorted({
        label
        for labels in diagnostic_labels_by_path.values()
        for label in labels
        if str(label).startswith('solved_')
    })
    policy_selected_paths = {
        policy['policy_name']: {
            _phase2h_path_key(item['path_vertex_ids']): item
            for item in policy['selected_candidates']
        }
        for policy in policies
    }
    selected_by_policy = {}
    still_open_by_policy = {}
    normalized_policy_coverage = {}
    for policy in policies:
        policy_name = policy['policy_name']
        selected_labels = []
        for label, path in open_by_label.items():
            selected = policy_selected_paths[policy_name].get(path)
            if selected is not None and not _phase2hr_is_baseline_candidate(selected):
                selected_labels.append(label)
        selected_labels = sorted(selected_labels, key=_phase2hr_open_label_sort_key)
        selected_by_policy[policy_name] = selected_labels
        still_open = [
            label for label in sorted(open_by_label, key=_phase2hr_open_label_sort_key)
            if label not in set(selected_labels)
        ]
        still_open_by_policy[policy_name] = still_open
        normalized_policy_coverage[policy_name] = {
            'count': len(selected_labels),
            'labels': selected_labels,
            'denominator': len(open_by_label),
        }
    open_details = []
    for label, path in PHASE2HR_CANONICAL_OPEN_RESIDUAL_PATHS:
        key = _phase2h_path_key(path)
        candidate = candidate_by_path.get(key)
        residual = residual_by_path.get(key, {})
        selected_normalized = {
            policy_name: (
                key in selected_paths and not _phase2hr_is_baseline_candidate(selected_paths[key])
            )
            for policy_name, selected_paths in policy_selected_paths.items()
        }
        skipped_normalized = {
            policy_name: not selected
            for policy_name, selected in selected_normalized.items()
        }
        reasons = {
            policy_name: (
                'selected_new_open_residual'
                if selected_normalized[policy_name]
                else _phase2hr_normalized_skip_reason(policy, label, key, candidate)
            )
            for policy in policies
            for policy_name in (policy['policy_name'],)
        }
        open_details.append(_phase2hr_open_residual_detail(
            label=label,
            aliases=next(group['aliases'] for group in alias_groups if group['canonical_label'] == label),
            path=key,
            candidate=candidate,
            residual=residual,
            selected_normalized=selected_normalized,
            skipped_normalized=skipped_normalized,
            reasons=reasons,
        ))
    already_marked_labels = set(solved_labels)
    for candidate in candidates:
        if _phase2hr_is_baseline_candidate(candidate):
            already_marked_labels.update(candidate['residual_match_labels'])
    legacy_label_space = {
        policy['policy_name']: {
            'count': policy.get('legacy_residual_coverage_count', len(policy.get('residual_paths_selected', ()))),
            'labels': list(policy.get('legacy_residual_coverage_labels', policy.get('residual_paths_selected', ()))),
        }
        for policy in policies
    }
    return {
        'canonical_open_residual_paths_total': len(PHASE2HR_CANONICAL_OPEN_RESIDUAL_PATHS),
        'canonical_open_residual_paths': [
            {'canonical_label': label, 'path_vertex_ids': list(_phase2h_path_key(path))}
            for label, path in PHASE2HR_CANONICAL_OPEN_RESIDUAL_PATHS
        ],
        'unique_open_residual_paths_selected_by_policy': selected_by_policy,
        'unique_open_residual_paths_still_open_by_policy': still_open_by_policy,
        'duplicate_label_count': duplicate_label_count,
        'solved_label_count': len(solved_labels),
        'already_marked_baseline_label_count': len(already_marked_labels),
        'residual_alias_groups': alias_groups,
        'legacy_label_space_coverage_by_policy': legacy_label_space,
        'normalized_open_residual_coverage_by_policy': normalized_policy_coverage,
        'open_residual_details': open_details,
    }


def _phase2hr_annotate_policies_with_normalized_metrics(policies, candidates, normalized):
    open_labels = {
        item['canonical_label']
        for item in normalized['canonical_open_residual_paths']
    }
    open_path_to_label = {
        tuple(item['path_vertex_ids']): item['canonical_label']
        for item in normalized['canonical_open_residual_paths']
    }
    for policy in policies:
        selected = policy['selected_candidates']
        baseline = [item for item in selected if _phase2hr_is_baseline_candidate(item)]
        new_selected = [item for item in selected if not _phase2hr_is_baseline_candidate(item)]
        new_open_labels = []
        for item in new_selected:
            label = open_path_to_label.get(_phase2h_path_key(item['path_vertex_ids']))
            if label is not None:
                new_open_labels.append(label)
        new_open_labels = sorted(set(new_open_labels), key=_phase2hr_open_label_sort_key)
        non_residual = [
            item for item in new_selected
            if _phase2h_path_key(item['path_vertex_ids']) not in open_path_to_label
        ]
        low_straightness = [
            item for item in new_selected
            if item.get('selected_low_straightness_two_edge_bridge', False)
        ]
        high_risk = [
            item for item in new_selected
            if item['candidate_class'] in PHASE2HR_HIGH_RISK_CLASSES
        ]
        duplicate_new = [
            item for item in new_selected
            if item['candidate_class'] == 'two_edge_duplicate_alternative'
        ]
        unsafe_new = [
            item for item in new_selected
            if item['candidate_class'] == 'non_original_or_missing_blender_edge'
        ]
        straight_but_tangent_weak = [
            item for item in new_selected
            if item.get('straight_but_tangent_weak', False)
        ]
        policy['legacy_selected_total'] = policy['selected_total']
        policy['selected_total_reported_legacy'] = policy['selected_total']
        policy['legacy_residual_coverage_labels'] = list(policy.get('residual_paths_selected', ()))
        policy['legacy_residual_coverage_count'] = len(policy['legacy_residual_coverage_labels'])
        policy['already_marked_baseline_selected'] = len(baseline)
        policy['newly_selected_total'] = len(new_selected)
        policy['newly_selected_by_class'] = _count_by_key(new_selected, 'candidate_class')
        policy['newly_selected_open_residual_labels'] = new_open_labels
        policy['newly_selected_non_residual_candidates'] = non_residual
        policy['unsafe_selected_total'] = len(unsafe_new)
        policy['duplicate_selected_total'] = len(duplicate_new)
        policy['selected_high_risk_class_total'] = len(high_risk)
        policy['selected_low_straightness_two_edge_bridge_total'] = len(low_straightness)
        policy['selected_straight_but_tangent_weak_total'] = len(straight_but_tangent_weak)
        policy['normalized_new_open_residual_coverage_count'] = len(new_open_labels)
        policy['normalized_new_open_residual_coverage_labels'] = new_open_labels
        decision = _phase2hr_policy_decision(policy)
        policy.update(decision)


def _phase2hr_policy_decision(policy):
    blocking = []
    if policy['normalized_new_open_residual_coverage_count'] == 0:
        blocking.append('normalized_open_residual_coverage_zero')
    if policy['already_marked_baseline_selected'] >= policy['legacy_selected_total']:
        blocking.append('coverage_inflated_by_already_marked_baseline')
    if policy['selected_low_straightness_two_edge_bridge_total'] > 0:
        blocking.append('low_straightness_endpoint_bridge_selected')
    if policy['selected_high_risk_class_total'] > 0:
        blocking.append('high_risk_class_selected')
    if policy['unsafe_selected_total'] > 0:
        blocking.append('unsafe_candidate_selected')
    if policy['duplicate_selected_total'] > 0:
        blocking.append('duplicate_candidate_selected')
    if policy['normalized_new_open_residual_coverage_count'] == 0:
        readiness = 'diagnostic_only'
    elif policy['unsafe_selected_total'] or policy['selected_high_risk_class_total']:
        readiness = 'diagnostic_only'
    elif policy['selected_low_straightness_two_edge_bridge_total']:
        readiness = 'candidate_for_visual_review'
    else:
        readiness = 'candidate_for_future_active_design'
    return {
        'high_risk_new_selection_count': policy['selected_high_risk_class_total'],
        'low_straightness_new_selection_count': policy['selected_low_straightness_two_edge_bridge_total'],
        'already_marked_baseline_count': policy['already_marked_baseline_selected'],
        'duplicate_new_selection_count': policy['duplicate_selected_total'],
        'unsafe_new_selection_count': policy['unsafe_selected_total'],
        'would_this_policy_be_safe_for_active_phase2j': bool(
            readiness == 'candidate_for_future_active_design' and not blocking
        ),
        'blocking_reasons': blocking,
        'production_readiness': readiness,
    }


def _phase2hr_normalized_summary(policies, normalized):
    coverage = normalized['normalized_open_residual_coverage_by_policy']
    ranked = sorted(
        (
            {
                'policy_name': policy_name,
                'normalized_new_open_residual_coverage_count': item['count'],
                'normalized_new_open_residual_coverage_labels': item['labels'],
            }
            for policy_name, item in coverage.items()
        ),
        key=lambda item: (-item['normalized_new_open_residual_coverage_count'], item['policy_name']),
    )
    best = ranked[0] if ranked else {
        'policy_name': None,
        'normalized_new_open_residual_coverage_count': 0,
        'normalized_new_open_residual_coverage_labels': [],
    }
    policies_with_low = [
        policy['policy_name'] for policy in policies
        if policy.get('selected_low_straightness_two_edge_bridge_total', 0) > 0
    ]
    policies_with_high_risk = [
        policy['policy_name'] for policy in policies
        if policy.get('selected_high_risk_class_total', 0) > 0
    ]
    ready = [
        policy['policy_name'] for policy in policies
        if policy.get('would_this_policy_be_safe_for_active_phase2j', False)
    ]
    straight_weak_count = sum(
        policy.get('selected_straight_but_tangent_weak_total', 0) for policy in policies
    )
    return {
        'canonical_open_residual_paths_total': normalized['canonical_open_residual_paths_total'],
        'best_policy_by_normalized_open_residual_coverage': best['policy_name'],
        'best_policy_new_open_residual_coverage_count': best['normalized_new_open_residual_coverage_count'],
        'best_policy_new_open_residual_coverage_labels': best['normalized_new_open_residual_coverage_labels'],
        'policies_ranked_by_normalized_coverage': ranked,
        'policies_with_low_straightness_selected': policies_with_low,
        'policies_with_high_risk_class_selected': policies_with_high_risk,
        'policies_production_ready': ready,
        'recommended_next_action_normalized': _phase2hr_recommended_next_action_normalized(
            policies_with_low,
            straight_weak_count,
            ready,
            best['normalized_new_open_residual_coverage_count'],
        ),
    }


def _phase2hr_recommended_next_action_normalized(policies_with_low, straight_weak_count, ready, best_count):
    if policies_with_low:
        return 'inspect_low_straightness_selected_candidates'
    if straight_weak_count:
        return 'inspect_straight_but_tangent_weak_candidates'
    if best_count > 0 and ready:
        return 'design_narrow_phase2j_from_normalized_policy'
    if best_count > 0:
        return 'refine_unified_scoring_before_active_repair'
    return 'no_active_repair_recommended'


def _phase2hr_open_residual_detail(
    *,
    label,
    aliases,
    path,
    candidate,
    residual,
    selected_normalized,
    skipped_normalized,
    reasons,
):
    return {
        'canonical_residual_label': label,
        'aliases': aliases,
        'path_vertex_ids': list(path),
        'current_class': residual.get('candidate_class_phase2e'),
        'current_status': residual.get('candidate_class_phase2e'),
        'current_rejection_or_skip_reason': (
            residual.get('phase_2b1_rejection_reason')
            or residual.get('skipped_reason')
            or residual.get('phase_2b_rejection_reason')
            or residual.get('phase_2a1_rejection_reason')
        ),
        'unified_candidate_class': None if candidate is None else candidate['candidate_class'],
        'all_edges_exist_in_blender': False if candidate is None else candidate['all_edges_exist_in_blender'],
        'edge_seam_flags_after_all_repairs': None if candidate is None else candidate['edge_seam_flags_after_all_repairs'],
        'already_all_marked': False if candidate is None else candidate['already_all_marked'],
        'selected_by_each_policy_normalized': selected_normalized,
        'skipped_by_each_policy_normalized': skipped_normalized,
        'reason_selected_or_skipped_by_policy': reasons,
        'path_straightness': None if candidate is None else candidate['path_straightness'],
        'min_endpoint_tangent_alignment': None if candidate is None else candidate['min_endpoint_tangent_alignment'],
        'endpoint_tangent_alignments': None if candidate is None else candidate['endpoint_tangent_alignments'],
        'safety_tier': None if candidate is None else candidate['safety_tier'],
        'loop_risk': None if candidate is None else candidate['loop_risk'],
        'topology_risk': None if candidate is None else candidate['topology_risk'],
        'would_require_new_repair_class': False if candidate is None else candidate['would_require_new_repair_class'],
        'would_require_cap_or_ranking_change': False if candidate is None else candidate['would_require_cap_or_ranking_change'],
        'would_require_topology_remapping': True if candidate is None else candidate['would_require_topology_remapping'],
    }


def _phase2hr_normalized_skip_reason(policy, label, path, candidate):
    selected = {
        _phase2h_path_key(item['path_vertex_ids']): item
        for item in policy['selected_candidates']
    }
    if path in selected and _phase2hr_is_baseline_candidate(selected[path]):
        return 'already_marked_baseline_not_new_gain'
    for class_items in policy.get('top_rejected_candidates_by_class', {}).values():
        for item in class_items:
            if _phase2h_path_key(item['path_vertex_ids']) == path:
                return item.get('simulated_skip_reason') or 'not_selected_by_simulation_policy'
    if candidate is None:
        return 'no_editable_candidate_for_open_residual'
    return 'not_selected_by_simulation_policy'


def _phase2hr_is_baseline_candidate(candidate):
    return bool(
        candidate.get('already_all_marked', False)
        or candidate.get('candidate_class') == 'current_selected_repair'
        or any(str(label).startswith('solved_') for label in candidate.get('residual_match_labels', ()))
    )


def _phase2hr_order_aliases(canonical_label, aliases):
    result = [canonical_label]
    residual_label = f'residual_{canonical_label}'
    if residual_label in aliases:
        result.append(residual_label)
    for label in sorted(aliases):
        if label not in result:
            result.append(label)
    return result


def _phase2hr_open_label_sort_key(label):
    order = {
        item_label: index
        for index, (item_label, _) in enumerate(PHASE2HR_CANONICAL_OPEN_RESIDUAL_PATHS)
    }
    return order.get(label, len(order))


PHASE2HR3_HIGH_RISK_CLASSES = {
    'two_edge_same_component_local_closure',
    'one_edge_endpoint_to_skeleton',
    'two_edge_endpoint_to_skeleton_or_near_junction',
    'three_edge_local_bridge',
    'two_edge_tangent_failed_endpoint_bridge',
    'non_original_or_missing_blender_edge',
}


def build_phase2h_r3_visual_review(simulation_payload):
    payload = simulation_payload or {}
    candidates_by_id, selected_policy_by_id = _phase2hr3_candidate_index(payload)
    selected_candidates = list(selected_policy_by_id)
    low_straightness = [
        _phase2hr3_low_straightness_report(candidates_by_id[candidate_id], policies)
        for candidate_id, policies in selected_policy_by_id.items()
        if _phase2hr3_is_low_straightness_selected(candidates_by_id[candidate_id])
    ]
    tangent_weak = [
        _phase2hr3_tangent_weak_report(candidate)
        for candidate in _phase2hr3_all_candidates(payload).values()
        if _phase2hr3_is_straight_but_tangent_weak(candidate)
    ]
    high_risk = [
        _phase2hr3_high_risk_report(candidates_by_id[candidate_id], policies, payload)
        for candidate_id, policies in selected_policy_by_id.items()
        if _phase2hr3_is_high_risk_probe_candidate(candidates_by_id[candidate_id], policies)
    ]
    rejected_tangent = _phase2hr3_rejected_tangent_failed_residuals(payload, selected_policy_by_id)
    counts = {
        'low_straightness_selected_candidates': len(low_straightness),
        'straight_but_tangent_weak_candidates': len(tangent_weak),
        'high_risk_probe_selected_candidates': len(high_risk),
        'rejected_tangent_failed_residuals': len(rejected_tangent),
    }
    decision_summary = _phase2hr3_decision_summary(
        payload,
        counts,
        low_straightness,
        tangent_weak,
        high_risk,
        rejected_tangent,
    )
    total_reported = sum(counts.values())
    return {
        'phase': '2H-R.3',
        'name': 'visual_review_local_continuity_candidates',
        'read_only': True,
        'seam_flags_unchanged': bool(payload.get('seam_flags_unchanged', True)),
        'not_applied_to_mesh': True,
        'probabilities_used': False,
        'diagnostic_paths_are_labels_only': True,
        'active_phase2j_allowed': False,
        'source_simulation_sidecar': 'prediction_unified_local_continuity_simulation_phase2h_r.json',
        'compact_sidecar': True,
        'low_straightness_selected_candidates': low_straightness,
        'straight_but_tangent_weak_candidates': tangent_weak,
        'high_risk_probe_selected_candidates': high_risk,
        'rejected_tangent_failed_residuals': rejected_tangent,
        'normalized_decision_summary': decision_summary,
        'compactness_summary': {
            'total_review_candidates_reported': total_reported,
            'counts_by_review_group': counts,
            'omitted_candidate_count_if_any': 0,
            'report_caps_if_any': {},
        },
    }


def _phase2hr3_candidate_index(payload):
    candidates = {}
    selected_policy_by_id = {}
    for policy in payload.get('policies', ()):
        policy_name = policy.get('policy_name')
        for candidate in policy.get('selected_candidates', ()):
            candidate_id = candidate.get('candidate_id')
            if candidate_id is None:
                continue
            candidates[candidate_id] = candidate
            selected_policy_by_id.setdefault(candidate_id, []).append(policy_name)
        for candidate in policy.get('residual_matched_candidates', ()):
            candidate_id = candidate.get('candidate_id')
            if candidate_id is not None:
                candidates.setdefault(candidate_id, candidate)
        for reports in policy.get('top_rejected_candidates_by_class', {}).values():
            for candidate in reports:
                candidate_id = candidate.get('candidate_id')
                if candidate_id is not None:
                    candidates.setdefault(candidate_id, candidate)
    for candidate in payload.get('straight_but_tangent_weak_candidates', ()):
        candidate_id = candidate.get('candidate_id')
        if candidate_id is not None:
            candidates.setdefault(candidate_id, candidate)
    return candidates, selected_policy_by_id


def _phase2hr3_all_candidates(payload):
    candidates, _ = _phase2hr3_candidate_index(payload)
    return candidates


def _phase2hr3_is_low_straightness_selected(candidate):
    return bool(
        candidate.get('candidate_class') == 'two_edge_inter_component_endpoint_bridge'
        and not candidate.get('already_all_marked', False)
        and candidate.get('path_straightness') is not None
        and candidate['path_straightness'] < 0.50
    )


def _phase2hr3_is_straight_but_tangent_weak(candidate):
    return bool(
        candidate.get('all_edges_exist_in_blender', True)
        and not candidate.get('already_all_marked', False)
        and candidate.get('path_straightness') is not None
        and candidate.get('min_endpoint_tangent_alignment') is not None
        and candidate['path_straightness'] >= 0.85
        and candidate['min_endpoint_tangent_alignment'] < 0.30
    )


def _phase2hr3_is_high_risk_probe_candidate(candidate, policies):
    return bool(
        candidate.get('candidate_class') in PHASE2HR3_HIGH_RISK_CLASSES
        and (
            'class_balanced_probe' in policies
            or 'conservative_length1_to_3' in policies
        )
    )


def _phase2hr3_low_straightness_report(candidate, selected_by_policies):
    report = _phase2hr3_common_candidate_report(candidate)
    severe = bool(candidate.get('path_straightness') is not None and candidate['path_straightness'] < 0.25)
    report.update({
        'selected_by_policies': list(selected_by_policies),
        'current_pipeline_rank_if_available': candidate.get('rank_v2_if_available'),
        'current_pipeline_skip_reason': candidate.get('simulated_skip_reason'),
        'low_straightness_warning': True,
        'severe_low_straightness_warning': severe,
        'visual_review_priority': 'highest' if severe else 'high',
        'active_repair_blocked': True,
        'recommended_architect_action': 'manual_visual_review_low_straightness_do_not_auto_select',
    })
    return report


def _phase2hr3_tangent_weak_report(candidate):
    report = _phase2hr3_common_candidate_report(candidate)
    report.update({
        'weak_endpoint_index': _phase2hr3_weak_endpoint_index(candidate),
        'seam_neighbor_used_for_tangent_if_available': None,
        'tangent_model_audit_needed': True,
        'do_not_select_automatically': True,
        'active_repair_blocked': True,
        'recommended_architect_action': 'audit_tangent_estimator_do_not_auto_select',
    })
    return report


def _phase2hr3_high_risk_report(candidate, selected_by_policies, payload):
    report = _phase2hr3_common_candidate_report(candidate)
    class_name = candidate.get('candidate_class')
    report.update({
        'candidate_class': class_name,
        'selected_by_policies': list(selected_by_policies),
        'candidate_pool_size_for_that_class': (
            payload.get('summary', {}).get('candidates_discovered_by_class', {}).get(class_name, 0)
        ),
        'would_require_new_repair_class': bool(candidate.get('would_require_new_repair_class', False)),
        'why_not_active_repair': _phase2hr3_high_risk_reason(class_name),
        'active_repair_blocked': True,
        'recommended_architect_action': _phase2hr3_high_risk_action(class_name),
    })
    return report


def _phase2hr3_rejected_tangent_failed_residuals(payload, selected_policy_by_id):
    details = payload.get('normalized_residual_coverage', {}).get('open_residual_details', ())
    reports = []
    candidate_by_path = {
        tuple(candidate.get('path_vertex_ids', ())): candidate
        for candidate in _phase2hr3_all_candidates(payload).values()
    }
    for detail in details:
        if detail.get('current_rejection_or_skip_reason') != 'tangent_alignment_failed':
            continue
        candidate = candidate_by_path.get(tuple(detail.get('path_vertex_ids', ())))
        base = _phase2hr3_common_candidate_report(candidate or detail)
        selected_by_probe = bool(
            candidate is not None
            and 'class_balanced_probe' in selected_policy_by_id.get(candidate.get('candidate_id'), ())
        )
        base.update({
            'residual_labels': list(detail.get('aliases', ())),
            'whether_one_sided_or_both_sided_tangent_failure': _phase2hr3_tangent_failure_sidedness(candidate),
            'whether_class_balanced_probe_selected_it': selected_by_probe,
            'reason_not_selected': None if selected_by_probe else _phase2hr3_detail_skip_reason(
                detail,
                'class_balanced_probe',
            ),
            'active_repair_blocked': True,
        })
        reports.append(base)
    return reports


def _phase2hr3_common_candidate_report(candidate):
    candidate = candidate or {}
    return {
        'path_vertex_ids': list(candidate.get('path_vertex_ids', ())),
        'path_edge_keys': list(candidate.get('path_edge_keys', ())),
        'blender_edge_indices': list(candidate.get('blender_edge_indices', ())),
        'residual_labels': list(candidate.get('residual_match_labels', ())),
        'residual_aliases': list(candidate.get('residual_match_labels', ())),
        'path_straightness': candidate.get('path_straightness'),
        'min_endpoint_tangent_alignment': candidate.get('min_endpoint_tangent_alignment'),
        'endpoint_tangent_alignments': candidate.get('endpoint_tangent_alignments'),
        'total_path_length': candidate.get('total_path_length'),
        'endpoint_distance': candidate.get('endpoint_distance'),
        'normalized_total_path_length': candidate.get('normalized_total_path_length_if_mesh_scale_available'),
        'normalized_endpoint_distance': candidate.get('normalized_endpoint_distance_if_mesh_scale_available'),
        'component_relation': candidate.get('component_relation'),
        'degree_pattern': list(candidate.get('degree_pattern', ())),
        'loop_risk': candidate.get('loop_risk'),
        'topology_risk': candidate.get('topology_risk'),
    }


def _phase2hr3_weak_endpoint_index(candidate):
    alignments = candidate.get('endpoint_tangent_alignments')
    if not alignments:
        return None
    values = [
        (index, value) for index, value in enumerate(alignments)
        if value is not None
    ]
    if not values:
        return None
    return min(values, key=lambda item: item[1])[0]


def _phase2hr3_tangent_failure_sidedness(candidate):
    if not candidate:
        return 'unknown'
    alignments = candidate.get('endpoint_tangent_alignments') or ()
    failed = [value for value in alignments if value is not None and value < 0.30]
    if len(failed) >= 2:
        return 'both_sided'
    if len(failed) == 1:
        return 'one_sided'
    return 'unknown'


def _phase2hr3_detail_skip_reason(detail, policy_name):
    reasons = detail.get('reason_selected_or_skipped_by_policy', {})
    return reasons.get(policy_name) or 'not_selected_by_simulation_policy'


def _phase2hr3_high_risk_reason(class_name):
    return {
        'two_edge_same_component_local_closure': 'same_component_closure_has_loop_risk',
        'one_edge_endpoint_to_skeleton': 'endpoint_to_skeleton_repair_not_designed',
        'two_edge_endpoint_to_skeleton_or_near_junction': 'endpoint_to_skeleton_or_junction_repair_not_designed',
        'three_edge_local_bridge': 'three_edge_repair_not_designed',
        'two_edge_tangent_failed_endpoint_bridge': 'tangent_failed_bridge_not_safe_for_auto_repair',
        'non_original_or_missing_blender_edge': 'candidate_is_not_editable_in_blender',
    }.get(class_name, 'high_risk_class_not_active_repair')


def _phase2hr3_high_risk_action(class_name):
    return {
        'two_edge_same_component_local_closure': 'review_loop_risk_do_not_auto_select',
        'one_edge_endpoint_to_skeleton': 'design_endpoint_to_skeleton_repair_before_selection',
        'two_edge_endpoint_to_skeleton_or_near_junction': 'design_endpoint_to_skeleton_repair_before_selection',
        'three_edge_local_bridge': 'design_three_edge_repair_before_selection',
        'two_edge_tangent_failed_endpoint_bridge': 'audit_tangent_failure_do_not_auto_select',
        'non_original_or_missing_blender_edge': 'inspect_topology_remapping_not_editable',
    }.get(class_name, 'manual_architect_review_required')


def _phase2hr3_decision_summary(
    simulation_payload,
    counts,
    low_straightness,
    tangent_weak,
    high_risk,
    rejected_tangent,
):
    normalized = simulation_payload.get('normalized_summary', {})
    coverage = simulation_payload.get('normalized_residual_coverage', {}).get(
        'normalized_open_residual_coverage_by_policy',
        {},
    )
    blocking_reasons = []
    if low_straightness:
        blocking_reasons.append('low_straightness_selected_candidates_require_manual_review')
    if tangent_weak:
        blocking_reasons.append('straight_but_tangent_weak_candidates_require_tangent_audit')
    if high_risk:
        blocking_reasons.append('high_risk_probe_candidates_require_new_repair_design')
    if rejected_tangent:
        blocking_reasons.append('tangent_failed_residuals_remain_rejected')
    return {
        'canonical_open_residual_paths_total': normalized.get('canonical_open_residual_paths_total', 0),
        'normalized_coverage_by_policy': coverage,
        'policies_production_ready': list(normalized.get('policies_production_ready', ())),
        'active_phase2j_allowed': False,
        'blocking_reasons': blocking_reasons,
        'low_straightness_selected_count': counts['low_straightness_selected_candidates'],
        'severe_low_straightness_selected_count': sum(
            1 for item in low_straightness if item['severe_low_straightness_warning']
        ),
        'straight_but_tangent_weak_count': counts['straight_but_tangent_weak_candidates'],
        'high_risk_probe_selected_count': counts['high_risk_probe_selected_candidates'],
        'rejected_tangent_failed_residual_count': counts['rejected_tangent_failed_residuals'],
        'recommended_next_action': 'manual_visual_review_low_straightness_and_tangent_audit',
    }


PHASE2JR_REPORT_CAPS = {
    'curved_two_edge_endpoint_bridge_candidates': 20,
    'straight_but_tangent_weak_audit_candidates': 20,
    'length_three_local_bridge_diagnostics': 20,
    'same_component_local_closure_diagnostics': 20,
}


PHASE2JR_REPRESENTATIVE_CASES = (
    (234, 319, 318, 214),
    (3098, 3185, 3192),
    (3098, 3097, 3192),
    (5477, 5520, 5483),
    (3783, 3784, 3753),
    (3783, 3750, 3753),
    (5562, 5464, 5553),
    (5562, 5463, 5553),
    (2804, 2709, 2815),
    (2804, 2816, 2815),
    (670, 669, 666),
    (670, 671, 666),
    (3006, 3008, 3039),
)


def simulate_phase2j_r_small_gap_rule(
    mesh,
    predicted_keys=None,
    local_repair_reports=(),
    two_edge_reports=(),
    endpoint_bridge_reports=(),
    residual_payload=None,
):
    seam_flags_before = tuple(bool(edge.use_seam) for edge in mesh.edges)
    reports, residual_by_path = _phase2h_collect_candidate_reports(
        mesh,
        predicted_keys=predicted_keys,
        local_repair_reports=local_repair_reports,
        two_edge_reports=two_edge_reports,
        endpoint_bridge_reports=endpoint_bridge_reports,
        residual_payload=residual_payload,
    )
    diagnostic_labels_by_path = _phase2hr_diagnostic_labels_by_path(residual_payload)
    candidates = [
        _phase2hr_candidate_for_simulation(report, diagnostic_labels_by_path)
        for report in reports
    ]
    candidates.sort(key=_phase2jr_sort_key)
    curved_all = [
        _phase2jr_curved_report(candidate)
        for candidate in candidates
        if _phase2jr_is_curved_two_edge_candidate(candidate)
    ]
    tangent_all = [
        _phase2jr_tangent_audit_report(candidate)
        for candidate in candidates
        if _phase2jr_is_straight_tangent_weak_candidate(candidate)
    ]
    length3_all = [
        _phase2jr_length3_report(candidate)
        for candidate in candidates
        if candidate.get('path_length_edges') == 3
    ]
    same_component_all = [
        _phase2jr_same_component_report(candidate)
        for candidate in candidates
        if (
            candidate.get('path_length_edges') == 2
            and candidate.get('component_relation') == 'same_component'
        )
    ]
    curved, curved_truncated = _phase2jr_cap_reports(
        curved_all,
        PHASE2JR_REPORT_CAPS['curved_two_edge_endpoint_bridge_candidates'],
    )
    tangent, tangent_truncated = _phase2jr_cap_reports(
        tangent_all,
        PHASE2JR_REPORT_CAPS['straight_but_tangent_weak_audit_candidates'],
    )
    length3, length3_truncated = _phase2jr_cap_reports(
        length3_all,
        PHASE2JR_REPORT_CAPS['length_three_local_bridge_diagnostics'],
    )
    same_component, same_component_truncated = _phase2jr_cap_reports(
        same_component_all,
        PHASE2JR_REPORT_CAPS['same_component_local_closure_diagnostics'],
    )
    all_by_path = {
        _phase2h_path_key(candidate['path_vertex_ids']): candidate
        for candidate in candidates
    }
    representative = _phase2jr_representative_case_matrix(all_by_path, residual_by_path)
    decision = _phase2jr_decision_summary(curved_all, tangent_all, length3_all, same_component_all)
    truncation = {
        'curved_two_edge_endpoint_bridge_candidates': curved_truncated,
        'straight_but_tangent_weak_audit_candidates': tangent_truncated,
        'length_three_local_bridge_diagnostics': length3_truncated,
        'same_component_local_closure_diagnostics': same_component_truncated,
        'representative_case_matrix': 0,
    }
    reported_counts = {
        'curved_two_edge_endpoint_bridge_candidates': len(curved),
        'straight_but_tangent_weak_audit_candidates': len(tangent),
        'length_three_local_bridge_diagnostics': len(length3),
        'same_component_local_closure_diagnostics': len(same_component),
        'representative_case_matrix': len(representative),
    }
    seam_flags_after = tuple(bool(edge.use_seam) for edge in mesh.edges)
    if seam_flags_after != seam_flags_before:
        raise RuntimeError('Phase 2J-R small-gap rule simulation modified seam flags.')
    return {
        'phase': '2J-R',
        'name': 'small_local_gap_closure_rule_simulation_v2',
        'read_only': True,
        'seam_flags_unchanged': True,
        'not_applied_to_mesh': True,
        'probabilities_used': False,
        'diagnostic_paths_are_labels_only': True,
        'active_phase2j_allowed': decision['active_phase2j_allowed'],
        'active_phase2j_allowed_means_review_only': True,
        'active_phase2j_blocking_reasons': list(decision['active_phase2j_blocking_reasons']),
        'recommended_next_action': decision['recommended_next_action'],
        'compact_sidecar': True,
        'curved_two_edge_endpoint_bridge_candidates': curved,
        'straight_but_tangent_weak_audit_candidates': tangent,
        'length_three_local_bridge_diagnostics': length3,
        'same_component_local_closure_diagnostics': same_component,
        'representative_case_matrix': representative,
        'decision_summary': decision,
        'compactness_summary': {
            'total_candidates_considered': len(candidates),
            'total_candidates_reported': sum(reported_counts.values()),
            'total_candidates_truncated': sum(truncation.values()),
            'per_group_reported_counts': reported_counts,
            'per_group_truncation_counts': truncation,
            'report_caps': dict(PHASE2JR_REPORT_CAPS),
        },
    }


def _phase2jr_sort_key(candidate):
    return (
        candidate.get('normalized_total_path_length_if_mesh_scale_available')
        if candidate.get('normalized_total_path_length_if_mesh_scale_available') is not None
        else 999.0,
        candidate.get('path_straightness') if candidate.get('path_straightness') is not None else 999.0,
        tuple(candidate.get('path_vertex_ids', ())),
    )


def _phase2jr_is_curved_two_edge_candidate(candidate):
    return bool(
        candidate.get('path_length_edges') == 2
        and candidate.get('all_edges_exist_in_blender', False)
        and not candidate.get('already_all_marked', False)
        and candidate.get('degree_pattern') == [1, 0, 1]
        and candidate.get('component_relation') == 'different_components'
        and candidate.get('loop_risk') == 'none'
        and candidate.get('topology_risk') == 'accepted_pattern'
        and _phase2jr_leq(candidate.get('normalized_total_path_length_if_mesh_scale_available'), 0.015)
        and _phase2jr_geq(candidate.get('min_endpoint_tangent_alignment'), 0.75)
        and candidate.get('path_straightness') is not None
        and candidate['path_straightness'] < 0.50
    )


def _phase2jr_is_straight_tangent_weak_candidate(candidate):
    alignments = candidate.get('endpoint_tangent_alignments') or ()
    values = [value for value in alignments if value is not None]
    return bool(
        candidate.get('path_length_edges') == 2
        and candidate.get('all_edges_exist_in_blender', False)
        and not candidate.get('already_all_marked', False)
        and candidate.get('component_relation') == 'different_components'
        and candidate.get('loop_risk') == 'none'
        and _phase2jr_geq(candidate.get('path_straightness'), 0.90)
        and _phase2jr_leq(candidate.get('normalized_total_path_length_if_mesh_scale_available'), 0.015)
        and values
        and max(values) >= 0.90
        and _phase2jr_lt(candidate.get('min_endpoint_tangent_alignment'), 0.30)
    )


def _phase2jr_curved_report(candidate):
    report = _phase2jr_common_report(candidate)
    report.update({
        'segment_length_ratio_if_computable': _phase2jr_segment_length_ratio(candidate),
        'turn_angle_degrees_if_computable': _phase2jr_turn_angle_degrees(candidate),
        'bend_angle_degrees_if_computable': _phase2jr_turn_angle_degrees(candidate),
        'current_rank_if_available': candidate.get('rank_v2_if_available'),
        'selected_by_existing_policy_if_available': bool(candidate.get('selected_by_simulation', False)),
        'curved_continuity_score': _phase2jr_curved_score(candidate),
        'strict_rule_passed': True,
        'would_be_active_candidate_under_strict_rule': True,
        'visual_confirmation_required': True,
        'why_selected_or_rejected': 'strict_curved_endpoint_bridge_rule_passed_for_review_only',
    })
    return report


def _phase2jr_tangent_audit_report(candidate):
    report = _phase2jr_common_report(candidate)
    audit_status = _phase2jr_tangent_audit_status(candidate)
    report.update({
        'weak_endpoint_index': _phase2hr3_weak_endpoint_index(candidate),
        'seam_neighbors_used_for_tangent_if_available': None,
        'alternative_seam_neighbor_tangents_if_computable': None,
        'component_direction_fallback_if_computable': None,
        'tangent_audit_status': audit_status,
        'would_be_active_candidate_under_strict_rule': audit_status == 'audit_supported_rescue',
        'visual_confirmation_required': True,
        'why_selected_or_rejected': (
            'straight_bridge_one_sided_tangent_weak_audit_supported'
            if audit_status == 'audit_supported_rescue'
            else 'straight_bridge_tangent_audit_not_supported'
        ),
    })
    return report


def _phase2jr_length3_report(candidate):
    return {
        'path_vertex_ids': list(candidate.get('path_vertex_ids', ())),
        'path_edge_keys': list(candidate.get('path_edge_keys', ())),
        'all_edges_exist_in_blender': bool(candidate.get('all_edges_exist_in_blender', False)),
        'component_relation': candidate.get('component_relation'),
        'degree_pattern': list(candidate.get('degree_pattern', ())),
        'endpoint_seam_vertex_flags': list(candidate.get('endpoint_seam_vertex_flags', ())),
        'intermediate_seam_vertex_flags': list(candidate.get('intermediate_seam_vertex_flags', ())),
        'would_create_loop': bool(candidate.get('would_create_loop', False)),
        'existing_seam_distance_between_endpoints_if_available': (
            candidate.get('existing_seam_distance_between_endpoints_if_available')
        ),
        'normalized_total_path_length': candidate.get('normalized_total_path_length_if_mesh_scale_available'),
        'path_straightness_if_computable': candidate.get('path_straightness'),
        'tangent_alignments_if_computable': candidate.get('endpoint_tangent_alignments'),
        'length3_active_safe': False,
        'blocking_reasons': ['length3_active_repair_not_implemented_in_phase2j_r'],
    }


def _phase2jr_same_component_report(candidate):
    return {
        'path_vertex_ids': list(candidate.get('path_vertex_ids', ())),
        'path_edge_keys': list(candidate.get('path_edge_keys', ())),
        'would_create_loop': bool(candidate.get('would_create_loop', False)),
        'existing_seam_distance_between_endpoints_if_available': (
            candidate.get('existing_seam_distance_between_endpoints_if_available')
        ),
        'normalized_total_path_length': candidate.get('normalized_total_path_length_if_mesh_scale_available'),
        'path_straightness': candidate.get('path_straightness'),
        'tangent_alignments': candidate.get('endpoint_tangent_alignments'),
        'same_component_active_safe': False,
        'blocking_reasons': ['same_component_active_repair_not_implemented_in_phase2j_r'],
    }


def _phase2jr_common_report(candidate):
    return {
        'path_vertex_ids': list(candidate.get('path_vertex_ids', ())),
        'path_length_edges': candidate.get('path_length_edges'),
        'path_edge_keys': list(candidate.get('path_edge_keys', ())),
        'blender_edge_indices': list(candidate.get('blender_edge_indices', ())),
        'all_edges_exist_in_blender': bool(candidate.get('all_edges_exist_in_blender', False)),
        'residual_labels_if_any': list(candidate.get('residual_match_labels', ())),
        'total_path_length': candidate.get('total_path_length'),
        'endpoint_distance': candidate.get('endpoint_distance'),
        'normalized_total_path_length': candidate.get('normalized_total_path_length_if_mesh_scale_available'),
        'normalized_endpoint_distance': candidate.get('normalized_endpoint_distance_if_mesh_scale_available'),
        'path_straightness': candidate.get('path_straightness'),
        'endpoint_tangent_alignments': candidate.get('endpoint_tangent_alignments'),
        'min_endpoint_tangent_alignment': candidate.get('min_endpoint_tangent_alignment'),
        'component_relation': candidate.get('component_relation'),
        'degree_pattern': list(candidate.get('degree_pattern', ())),
        'loop_risk': candidate.get('loop_risk'),
        'topology_risk': candidate.get('topology_risk'),
        'duplicate_endpoint_pair_key': list(candidate.get('duplicate_endpoint_pair_key', ())),
    }


def _phase2jr_representative_case_matrix(candidate_by_path, residual_by_path):
    rows = []
    for path in PHASE2JR_REPRESENTATIVE_CASES:
        key = _phase2h_path_key(path)
        candidate = candidate_by_path.get(key)
        residual = residual_by_path.get(key, {})
        group = _phase2jr_group_for_candidate(candidate)
        rows.append({
            'path_vertex_ids': list(path),
            'found_in_candidate_space': candidate is not None,
            'found_in_current_residual_space': bool(residual),
            'all_edges_exist_in_blender': None if candidate is None else candidate.get('all_edges_exist_in_blender'),
            'current_class': residual.get('candidate_class_phase2e') or (
                None if candidate is None else candidate.get('candidate_class')
            ),
            'proposed_phase2j_r_group': group,
            'active_safe_under_strict_rule': _phase2jr_candidate_active_safe(candidate),
            'visual_confirmation_required': bool(candidate is not None and _phase2jr_candidate_active_safe(candidate)),
            'blocking_reasons': _phase2jr_representative_blocking_reasons(candidate, group),
            'metrics': {} if candidate is None else {
                'normalized_total_path_length': candidate.get('normalized_total_path_length_if_mesh_scale_available'),
                'path_straightness': candidate.get('path_straightness'),
                'min_endpoint_tangent_alignment': candidate.get('min_endpoint_tangent_alignment'),
                'endpoint_tangent_alignments': candidate.get('endpoint_tangent_alignments'),
            },
            'possible_canonical_match': _phase2jr_possible_canonical_match(path, candidate_by_path),
        })
    return rows


def _phase2jr_decision_summary(curved, tangent, length3, same_component):
    active_candidates = [
        report for report in list(curved) + list(tangent)
        if report.get('would_be_active_candidate_under_strict_rule', False)
    ]
    residual_matched = [
        report for report in active_candidates
        if report.get('residual_labels_if_any')
    ]
    blocking = []
    if len(residual_matched) < 2:
        blocking.append('fewer_than_two_residual_matched_strict_candidates')
    if len(active_candidates) > 3:
        blocking.append('simulation_cap_exceeded_more_than_three_candidates')
    if any(report.get('path_length_edges') != 2 for report in active_candidates):
        blocking.append('non_length2_candidate_would_be_active')
    if any(report.get('component_relation') != 'different_components' for report in active_candidates):
        blocking.append('non_inter_component_candidate_would_be_active')
    if any(report.get('loop_risk') != 'none' for report in active_candidates):
        blocking.append('loop_risk_candidate_would_be_active')
    if any(not report.get('visual_confirmation_required', False) for report in active_candidates):
        blocking.append('strict_candidate_missing_visual_confirmation_requirement')
    active_allowed = not blocking
    return {
        'strict_curved_candidates_total': len(curved),
        'strict_curved_candidates_residual_matched': sum(
            1 for report in curved if report.get('residual_labels_if_any')
        ),
        'strict_curved_candidates_non_residual': sum(
            1 for report in curved if not report.get('residual_labels_if_any')
        ),
        'straight_tangent_audit_candidates_total': len(tangent),
        'audit_supported_rescue_count': sum(
            1 for report in tangent
            if report.get('tangent_audit_status') == 'audit_supported_rescue'
        ),
        'length3_diagnostic_count': len(length3),
        'same_component_diagnostic_count': len(same_component),
        'active_phase2j_allowed': active_allowed,
        'active_phase2j_allowed_means_review_only': True,
        'active_phase2j_blocking_reasons': blocking,
        'recommended_next_action': (
            'architect_review_strict_phase2j_candidates'
            if active_allowed
            else 'no_active_phase2j_repair_recommended_from_simulation'
        ),
    }


def _phase2jr_candidate_active_safe(candidate):
    if candidate is None:
        return False
    return bool(
        _phase2jr_is_curved_two_edge_candidate(candidate)
        or (
            _phase2jr_is_straight_tangent_weak_candidate(candidate)
            and _phase2jr_tangent_audit_status(candidate) == 'audit_supported_rescue'
        )
    )


def _phase2jr_group_for_candidate(candidate):
    if candidate is None:
        return None
    if _phase2jr_is_curved_two_edge_candidate(candidate):
        return 'curved_two_edge_endpoint_bridge_candidates'
    if _phase2jr_is_straight_tangent_weak_candidate(candidate):
        return 'straight_but_tangent_weak_audit_candidates'
    if candidate.get('path_length_edges') == 3:
        return 'length_three_local_bridge_diagnostics'
    if candidate.get('path_length_edges') == 2 and candidate.get('component_relation') == 'same_component':
        return 'same_component_local_closure_diagnostics'
    return 'not_in_phase2j_r_scope'


def _phase2jr_representative_blocking_reasons(candidate, group):
    if candidate is None:
        return ['representative_path_not_found_in_candidate_space']
    if not candidate.get('all_edges_exist_in_blender', False):
        return ['missing_or_non_original_edges_not_active_safe']
    if group == 'length_three_local_bridge_diagnostics':
        return ['length3_active_repair_not_implemented_in_phase2j_r']
    if group == 'same_component_local_closure_diagnostics':
        return ['same_component_active_repair_not_implemented_in_phase2j_r']
    if _phase2jr_candidate_active_safe(candidate):
        return ['visual_confirmation_required_before_any_future_active_repair']
    return ['strict_phase2j_r_rule_not_passed']


def _phase2jr_possible_canonical_match(path, candidate_by_path):
    key = _phase2h_path_key(path)
    if key in candidate_by_path:
        return None
    reverse = _phase2h_path_key(tuple(reversed(path)))
    if reverse in candidate_by_path:
        return list(reverse)
    return None


def _phase2jr_cap_reports(reports, cap):
    reports = sorted(reports, key=lambda report: tuple(report.get('path_vertex_ids', ())))
    return reports[:cap], max(0, len(reports) - cap)


def _phase2jr_curved_score(candidate):
    return [
        -float(candidate.get('min_endpoint_tangent_alignment') or 0.0),
        float(candidate.get('path_straightness') or 0.0),
        float(candidate.get('normalized_total_path_length_if_mesh_scale_available') or 999.0),
        list(candidate.get('path_vertex_ids', ())),
    ]


def _phase2jr_segment_length_ratio(candidate):
    total = candidate.get('total_path_length')
    endpoint = candidate.get('endpoint_distance')
    if total in (None, 0) or endpoint is None:
        return None
    return endpoint / total


def _phase2jr_turn_angle_degrees(candidate):
    straightness = candidate.get('path_straightness')
    if straightness is None:
        return None
    clamped = max(-1.0, min(1.0, float(straightness)))
    return math.degrees(math.acos(clamped))


def _phase2jr_tangent_audit_status(candidate):
    alignments = [
        value for value in (candidate.get('endpoint_tangent_alignments') or ())
        if value is not None
    ]
    if not alignments:
        return 'audit_inconclusive'
    if (
        len(alignments) >= 2
        and max(alignments) >= 0.95
        and min(alignments) >= 0.0
        and _phase2jr_geq(candidate.get('path_straightness'), 0.95)
    ):
        return 'audit_supported_rescue'
    if min(alignments) < 0.0:
        return 'audit_reject'
    return 'audit_inconclusive'


def _phase2jr_leq(value, threshold):
    return value is not None and value <= threshold


def _phase2jr_lt(value, threshold):
    return value is not None and value < threshold


def _phase2jr_geq(value, threshold):
    return value is not None and value >= threshold


def _safe_ratio(value, denominator):
    if value is None or denominator in (None, 0):
        return None
    return value / denominator


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
        'blender_two_edge_endpoint_bridge_raw_allowed_total': (
            result.blender_two_edge_endpoint_bridge_raw_allowed_total
        ),
        'blender_two_edge_endpoint_bridge_deduplicated_allowed_total': (
            result.blender_two_edge_endpoint_bridge_deduplicated_allowed_total
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
        'blender_two_edge_endpoint_bridge_duplicate_endpoint_pairs_suppressed': (
            result.blender_two_edge_endpoint_bridge_duplicate_endpoint_pairs_suppressed
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
        'blender_two_edge_endpoint_bridge_added_candidate_due_to_cap_increase': (
            result.blender_two_edge_endpoint_bridge_added_candidate_due_to_cap_increase
        ),
        'blender_two_edge_endpoint_bridge_previous_rank_9_selected': (
            result.blender_two_edge_endpoint_bridge_previous_rank_9_selected
        ),
        'blender_two_edge_endpoint_bridge_selected_rank_9_candidate': (
            result.blender_two_edge_endpoint_bridge_selected_rank_9_candidate
        ),
        'blender_curved_two_edge_endpoint_bridge_enabled': (
            result.blender_curved_two_edge_endpoint_bridge_enabled
        ),
        'blender_curved_two_edge_endpoint_bridge_candidates_total': (
            result.blender_curved_two_edge_endpoint_bridge_candidates_total
        ),
        'blender_curved_two_edge_endpoint_bridge_eligible_total': (
            result.blender_curved_two_edge_endpoint_bridge_eligible_total
        ),
        'blender_curved_two_edge_endpoint_bridge_paths_marked': (
            result.blender_curved_two_edge_endpoint_bridge_paths_marked
        ),
        'blender_curved_two_edge_endpoint_bridge_edges_marked': (
            result.blender_curved_two_edge_endpoint_bridge_edges_marked
        ),
        'blender_curved_two_edge_endpoint_bridge_over_cap': (
            result.blender_curved_two_edge_endpoint_bridge_over_cap
        ),
        'blender_curved_two_edge_endpoint_bridge_safety_cap': (
            result.blender_curved_two_edge_endpoint_bridge_safety_cap
        ),
        'blender_curved_two_edge_endpoint_bridge_candidate_reports': list(
            result.blender_curved_two_edge_endpoint_bridge_candidate_reports
        ),
        'selected_curved_two_edge_endpoint_bridge_reports': list(
            result.selected_curved_two_edge_endpoint_bridge_reports
        ),
        'rejected_curved_two_edge_endpoint_bridge_reports': list(
            result.rejected_curved_two_edge_endpoint_bridge_reports
        ),
        'probabilities_used_for_curved_repair': result.probabilities_used_for_curved_repair,
        'phase2j_curved_repair_uses_hardcoded_paths': (
            result.phase2j_curved_repair_uses_hardcoded_paths
        ),
        'phase2j_curved_repair_uses_probabilities': (
            result.phase2j_curved_repair_uses_probabilities
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
        'production_hardcoded_path_exceptions_enabled': (
            result.production_hardcoded_path_exceptions_enabled
        ),
        'production_hardcoded_path_exceptions_removed': (
            result.production_hardcoded_path_exceptions_removed
        ),
        'diagnostic_path_labels_read_only': result.diagnostic_path_labels_read_only,
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


def write_residual_gap_phase2e_debug(json_path, result):
    debug_path = json_path.rsplit('.', 1)[0] + '_residual_gap_phase2e_debug.json'
    payload = result.residual_gap_phase2e_debug or {
        'summary': {
            'residual_paths_total': 0,
            'recommended_next_action': 'no_dominant_next_action',
        },
        'paths': [],
        'read_only': True,
    }
    with open(debug_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    return debug_path


def format_residual_gap_phase2e_summary(payload, debug_path):
    summary = payload.get('summary', {})
    return (
        f"Phase 2E residual debug: {summary.get('residual_paths_total', 0)} residual paths, "
        f"{summary.get('residual_paths_already_all_marked', 0)} already marked, "
        f"{summary.get('residual_paths_phase_2b1_rank_below_cap', 0)} rank-below-cap, "
        f"{summary.get('residual_paths_phase_2b1_duplicate_suppressed', 0)} duplicate-suppressed, "
        f"{summary.get('residual_paths_phase_2b1_tangent_failed', 0)} tangent-failed, "
        f"{summary.get('residual_paths_three_edge_local_bridge', 0)} three-edge, "
        f"{summary.get('residual_paths_endpoint_to_skeleton_or_near_junction', 0)} endpoint-to-skeleton, "
        f"{summary.get('residual_paths_same_component_two_edge_local_bridge', 0)} same-component, "
        f"{summary.get('residual_paths_non_original_or_missing_blender_edge', 0)} missing-edge. "
        f"Recommended next action: {summary.get('recommended_next_action', 'no_dominant_next_action')}. "
        f"Sidecar: {debug_path}"
    )


def build_rank_9_to_16_review(result):
    allowed = [
        dict(report)
        for report in result.blender_two_edge_endpoint_bridge_allowed_candidate_reports
    ]
    allowed.sort(key=lambda report: report.get('rank_v2') or 10**9)
    residual_by_path = _residual_phase2e_reports_by_path(result.residual_gap_phase2e_debug)
    cap_selections = {
        cap: _rank_review_simulated_selection(allowed, cap)
        for cap in (8, 9, 10, 12, 16)
    }
    current_cap = int(result.blender_two_edge_endpoint_bridge_safety_cap)
    current_selected = cap_selections.get(current_cap, cap_selections[9])
    current_keys = {_rank_review_path_key(report) for report in current_selected}
    review_reports = [
        _rank_review_candidate_report(report, residual_by_path, cap_selections)
        for report in allowed
        if 9 <= int(report.get('rank_v2') or 0) <= 16
    ]
    cap_summaries = [
        _rank_review_cap_summary(cap, cap_selections[cap], current_keys, current_cap)
        for cap in (8, 9, 10, 12, 16)
    ]
    special = _rank_review_special_path_report(review_reports, (5149, 3003, 3005))
    summary = _rank_review_summary(result, review_reports, cap_summaries)
    return {
        'summary': summary,
        'rank_9_to_16_candidates': review_reports,
        'hypothetical_cap_summaries': cap_summaries,
        'special_reports': {
            'path_5149_3003_3005': special,
        },
        'rank_9_to_16_debug_edge_indices_by_rank': {
            str(report['rank_v2']): report['path_edge_indices_blender']
            for report in review_reports
        },
        'read_only': True,
    }


def write_rank_9_to_16_review(json_path, result):
    debug_path = json_path.rsplit('.', 1)[0] + '_rank_9_to_16_review.json'
    payload = build_rank_9_to_16_review(result)
    with open(debug_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    return debug_path


def format_rank_9_to_16_review_summary(payload, debug_path):
    summary = payload.get('summary', {})
    special = payload.get('special_reports', {}).get('path_5149_3003_3005', {})
    path_status = 'not found'
    if special.get('found_in_review'):
        path_status = (
            f"rank {special.get('rank_v2')} "
            f"{'cap9-selectable' if special.get('would_be_selected_if_cap_9') else 'not-cap9-selectable'}"
        )
    return (
        f"Phase 2F rank review: ranks {summary.get('review_rank_start', 9)}-"
        f"{summary.get('review_rank_end', 16)} analyzed, "
        f"rank 9 candidate status for [5149,3003,3005]: {path_status}, "
        f"cap=9 would add "
        f"{_rank_review_cap_added_count(payload, 9)} candidate(s), "
        f"recommended_next_action={summary.get('recommended_next_action', 'keep_cap_8')}. "
        f"Sidecar: {debug_path}"
    )


def write_general_residual_candidates_phase2h(json_path, result):
    debug_path = json_path.rsplit('.', 1)[0] + '_general_residual_candidates_phase2h.json'
    payload = result.general_residual_candidates_phase2h or {
        'summary': {
            'total_candidates_discovered_before_truncation': 0,
            'recommended_next_action': 'review_general_candidate_distribution',
        },
        'candidates': [],
        'human_residual_mapping': [],
        'read_only': True,
    }
    with open(debug_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    return debug_path


def write_unified_local_continuity_simulation_phase2h_r(json_path, result):
    debug_path = json_path.rsplit('.', 1)[0] + '_unified_local_continuity_simulation_phase2h_r.json'
    payload = result.unified_local_continuity_simulation_phase2h_r or {
        'phase': '2H-R.1',
        'name': 'unified_local_continuity_selector_simulation',
        'summary': {
            'recommended_next_action': 'defer_due_to_unsafe_candidate_distribution',
            'candidates_selected_by_policy': {},
            'residual_coverage_by_policy': {},
        },
        'policies': [],
        'residual_coverage': [],
        'read_only': True,
        'seam_flags_unchanged': True,
        'not_applied_to_mesh': True,
        'probabilities_used': False,
        'diagnostic_paths_are_labels_only': True,
        'compact_sidecar': True,
    }
    with open(debug_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    return debug_path


def write_phase2h_r3_visual_review(json_path, result):
    debug_path = json_path.rsplit('.', 1)[0] + '_phase2h_r3_visual_review.json'
    payload = result.phase2h_r3_visual_review or build_phase2h_r3_visual_review(
        result.unified_local_continuity_simulation_phase2h_r or {}
    )
    with open(debug_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    return debug_path


def write_phase2j_r_small_gap_rule_simulation(json_path, result):
    debug_path = json_path.rsplit('.', 1)[0] + '_phase2j_r_small_gap_rule_simulation.json'
    payload = result.phase2j_r_small_gap_rule_simulation or {
        'phase': '2J-R',
        'name': 'small_local_gap_closure_rule_simulation_v2',
        'read_only': True,
        'seam_flags_unchanged': True,
        'not_applied_to_mesh': True,
        'probabilities_used': False,
        'diagnostic_paths_are_labels_only': True,
        'active_phase2j_allowed': False,
        'active_phase2j_allowed_means_review_only': True,
        'active_phase2j_blocking_reasons': ['simulation_not_available'],
        'recommended_next_action': 'no_active_phase2j_repair_recommended_from_simulation',
        'compact_sidecar': True,
    }
    with open(debug_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    return debug_path


def format_phase2j_r_small_gap_rule_simulation_summary(payload, debug_path):
    decision = payload.get('decision_summary', {})
    return (
        f"Phase 2J-R small-gap simulation: "
        f"curved={decision.get('strict_curved_candidates_total', 0)}, "
        f"tangent_audit={decision.get('straight_tangent_audit_candidates_total', 0)}, "
        f"length3={decision.get('length3_diagnostic_count', 0)}, "
        f"same_component={decision.get('same_component_diagnostic_count', 0)}, "
        f"active_phase2j_allowed={str(payload.get('active_phase2j_allowed', False)).lower()}. "
        f"Sidecar: {debug_path}"
    )


def format_phase2h_r3_visual_review_summary(payload, debug_path):
    summary = payload.get('normalized_decision_summary', {})
    return (
        f"Phase 2H-R.3 visual review: "
        f"low_straightness={summary.get('low_straightness_selected_count', 0)}, "
        f"straight_but_tangent_weak={summary.get('straight_but_tangent_weak_count', 0)}, "
        f"high_risk_probe={summary.get('high_risk_probe_selected_count', 0)}, "
        f"active_phase2j_allowed={str(payload.get('active_phase2j_allowed', False)).lower()}. "
        f"Sidecar: {debug_path}"
    )


def format_unified_local_continuity_simulation_phase2h_r_summary(payload, debug_path):
    summary = payload.get('summary', {})
    residual_coverage = summary.get('residual_coverage_by_policy', {})
    best_coverage = max(residual_coverage.values(), default=0)
    residual_total = len(payload.get('residual_coverage', ()))
    return (
        f"Unified continuity simulation: "
        f"policies={len(payload.get('policies', ()))}, "
        f"candidates considered={payload.get('total_candidates_considered', 0)}, "
        f"reported={payload.get('total_candidates_reported', 0)}, "
        f"residual coverage best={best_coverage}/{residual_total}, "
        f"recommended_next_action={summary.get('recommended_next_action', 'defer_due_to_unsafe_candidate_distribution')}. "
        f"Sidecar: {debug_path}"
    )


def format_general_residual_candidates_phase2h_summary(payload, debug_path):
    summary = payload.get('summary', {})
    by_length = summary.get('candidates_by_path_length', {})
    coverage = summary.get('residual_coverage_by_class', {})
    return (
        f"Phase 2H candidate collector: "
        f"{summary.get('total_candidates_stored_after_truncation', 0)} stored / "
        f"{summary.get('total_candidates_discovered_before_truncation', 0)} discovered local candidates, "
        f"length1={by_length.get(1, by_length.get('1', 0))}, "
        f"length2={by_length.get(2, by_length.get('2', 0))}, "
        f"length3={by_length.get(3, by_length.get('3', 0))} discovered, "
        f"truncated={summary.get('total_candidates_truncated', 0)}. "
        f"Residual coverage: {coverage}. "
        f"Recommended next action: {summary.get('recommended_next_action', 'review_general_candidate_distribution')}. "
        f"Sidecar: {debug_path}"
    )


def _residual_phase2e_reports_by_path(residual_payload):
    result = {}
    for report in (residual_payload or {}).get('paths', []):
        path = report.get('path_vertex_ids')
        if isinstance(path, list) and len(path) == 3:
            result[_canonical_two_edge_path(path)] = report
    return result


def _rank_review_simulated_selection(allowed, cap):
    selected = []
    reserved_edges = set()
    for report in allowed:
        if report.get('duplicate_endpoint_pair_suppressed', False):
            continue
        path_edges = {
            tuple(edge_key)
            for edge_key in report.get('path_edge_keys', [])
            if isinstance(edge_key, list) and len(edge_key) == 2
        }
        if path_edges.intersection(reserved_edges):
            continue
        selected.append(report)
        reserved_edges.update(path_edges)
        if len(selected) >= int(cap):
            break
    return selected


def _rank_review_candidate_report(report, residual_by_path, cap_selections):
    path = tuple(report['path_vertex_ids'])
    residual = residual_by_path.get(_canonical_two_edge_path(path), {})
    weak = _rank_review_is_weak_geometry(report)
    strong = _rank_review_is_strong_geometry(report)
    human_labels = list(report.get('human_gap_match_labels', []))
    residual_label = residual.get('label')
    is_residual = bool(residual)
    review_class = _rank_review_candidate_class(report, residual, weak, strong)
    cap_flags = {
        cap: _rank_review_path_key(report) in {
            _rank_review_path_key(selected) for selected in cap_selections[cap]
        }
        for cap in (9, 10, 12, 16)
    }
    return {
        'rank': report.get('rank_v2'),
        'rank_v2': report.get('rank_v2'),
        'path_vertex_ids': list(report['path_vertex_ids']),
        'path_edge_keys': [list(edge_key) for edge_key in report.get('path_edge_keys', [])],
        'path_edge_indices_blender': list(report.get('path_edge_indices_blender', [])),
        'endpoint_pair_key': list(report.get('endpoint_pair_key', [])),
        'selected_for_marking': bool(report.get('selected_for_marking', False)),
        'marked': bool(report.get('marked', False)),
        'skipped_reason': report.get('skipped_reason'),
        'duplicate_endpoint_pair_suppressed': bool(report.get('duplicate_endpoint_pair_suppressed', False)),
        'conflict_reason': report.get('conflict_reason'),
        'continuity_tier': report.get('continuity_tier'),
        'q_floor': report.get('q_floor'),
        'q_sum': report.get('q_sum'),
        'total_path_length': report.get('total_path_length'),
        'endpoint_distance': report.get('endpoint_distance'),
        'path_straightness': report.get('path_straightness'),
        'endpoint_tangent_alignment_u': report.get('endpoint_tangent_alignment_u'),
        'endpoint_tangent_alignment_v': report.get('endpoint_tangent_alignment_v'),
        'min_endpoint_tangent_alignment': report.get('min_endpoint_tangent_alignment'),
        'degree_pattern': report.get('degree_pattern'),
        'component_ids_before': list(report.get('component_ids_before', [])),
        'human_gap_match_labels': human_labels,
        'old_validation_target_match_label': report.get('old_validation_target_match_label'),
        'is_residual_human_path': is_residual,
        'residual_phase2e_class_if_available': residual.get('candidate_class_phase2e'),
        'residual_recommended_followup_if_available': residual.get('recommended_followup'),
        'candidate_review_class': review_class,
        'would_be_selected_if_cap_9': cap_flags[9],
        'would_be_selected_if_cap_10': cap_flags[10],
        'would_be_selected_if_cap_12': cap_flags[12],
        'would_be_selected_if_cap_16': cap_flags[16],
        'cap_increase_risk': _rank_review_candidate_risk(report, weak, strong),
        'visual_review_priority': _rank_review_visual_priority(report, residual, weak, strong),
        'rank_delta_from_cap': None if report.get('rank_v2') is None else int(report['rank_v2']) - 9,
        'residual_label_if_available': residual_label,
    }


def _rank_review_candidate_class(report, residual, weak, strong):
    if report.get('duplicate_endpoint_pair_suppressed', False):
        return 'duplicate_alternative'
    if weak:
        return 'weak_geometry_rank_below_cap'
    if residual and strong:
        return 'strong_human_rank_below_cap'
    if not residual and not report.get('human_gap_match_labels'):
        return 'non_human_rank_below_cap'
    if report.get('marked', False) and residual:
        return 'backend_bridge_apply_mismatch_candidate'
    return 'unknown_review_candidate'


def _rank_review_candidate_risk(report, weak, strong):
    if report.get('duplicate_endpoint_pair_suppressed', False):
        return 'duplicate_suppressed'
    if weak:
        return 'not_recommended'
    if strong:
        return 'low'
    return 'needs_visual_review'


def _rank_review_visual_priority(report, residual, weak, strong):
    if report.get('duplicate_endpoint_pair_suppressed', False):
        return 'low'
    if residual and strong:
        return 'high'
    if residual and not weak:
        return 'medium'
    if weak:
        return 'low'
    return 'medium'


def _rank_review_is_weak_geometry(report):
    tier = report.get('continuity_tier')
    straightness = report.get('path_straightness')
    q_floor = report.get('q_floor')
    return bool(
        (tier is not None and tier >= 3)
        or (straightness is not None and straightness < 0.5)
        or (q_floor is not None and q_floor < 0.5)
    )


def _rank_review_is_strong_geometry(report):
    tier = report.get('continuity_tier')
    straightness = report.get('path_straightness')
    q_floor = report.get('q_floor')
    q_sum = report.get('q_sum')
    return bool(
        tier is not None
        and tier <= 1
        and q_floor is not None
        and q_floor >= 0.7
        and q_sum is not None
        and q_sum >= 1.4
        and straightness is not None
        and straightness >= 0.7
    )


def _rank_review_cap_summary(cap, selected_for_cap, current_keys, current_cap):
    added = [
        report for report in selected_for_cap
        if _rank_review_path_key(report) not in current_keys
    ]
    weak_added = [report for report in added if _rank_review_is_weak_geometry(report)]
    human_added = [
        report for report in added
        if report.get('human_gap_match_labels')
    ]
    duplicate_added = [
        report for report in added
        if report.get('duplicate_endpoint_pair_suppressed', False)
    ]
    non_human_added = [
        report for report in added
        if not report.get('human_gap_match_labels')
    ]
    return {
        'hypothetical_cap': int(cap),
        'additional_candidates_selected': len(added),
        'additional_human_candidates_selected': len(human_added),
        'additional_duplicate_candidates_selected': len(duplicate_added),
        'additional_non_human_candidates_selected': len(non_human_added),
        'min_continuity_tier_added': _min_report_value(added, 'continuity_tier'),
        'min_q_floor_added': _min_report_value(added, 'q_floor'),
        'min_q_sum_added': _min_report_value(added, 'q_sum'),
        'min_path_straightness_added': _min_report_value(added, 'path_straightness'),
        'candidate_labels_added': [_rank_review_candidate_label(report) for report in added],
        'path_vertex_ids_added': [list(report['path_vertex_ids']) for report in added],
        'risk_summary': _rank_review_cap_risk(cap, added, weak_added, human_added, current_cap),
    }


def _rank_review_cap_risk(cap, added, weak_added, human_added, current_cap):
    if int(cap) == int(current_cap):
        return 'current'
    if not added:
        return 'current'
    if len(weak_added) >= max(1, len(added) // 2):
        return 'not_recommended'
    if int(cap) == 9 and len(added) == 1 and len(human_added) == 1:
        return 'low'
    return 'needs_visual_review'


def _rank_review_summary(result, review_reports, cap_summaries):
    human = [report for report in review_reports if report['is_residual_human_path']]
    duplicates = [report for report in review_reports if report['duplicate_endpoint_pair_suppressed']]
    weak = [report for report in review_reports if _rank_review_report_is_weak(report)]
    strong = [report for report in review_reports if _rank_review_report_is_strong(report)]
    summary = {
        'selection_policy': result.blender_two_edge_endpoint_bridge_selection_policy,
        'current_safety_cap': result.blender_two_edge_endpoint_bridge_safety_cap,
        'selected_rank_threshold': result.blender_two_edge_endpoint_bridge_selected_rank_threshold,
        'raw_allowed_total': result.blender_two_edge_endpoint_bridge_raw_allowed_total,
        'deduplicated_allowed_total': result.blender_two_edge_endpoint_bridge_deduplicated_allowed_total,
        'selected_total': result.blender_two_edge_endpoint_bridge_paths_marked,
        'review_rank_start': 9,
        'review_rank_end': 16,
        'reviewed_candidate_count': len(review_reports),
        'human_matched_review_candidates': len(human),
        'duplicate_suppressed_review_candidates': len(duplicates),
        'non_human_review_candidates': len(review_reports) - len(human),
        'weak_geometry_review_candidates': len(weak),
        'strong_geometry_review_candidates': len(strong),
    }
    summary['recommended_next_action'] = _rank_review_recommendation(review_reports, cap_summaries)
    return summary


def _rank_review_recommendation(review_reports, cap_summaries):
    rank_9 = next((report for report in review_reports if report['rank_v2'] == 9), None)
    cap_9 = next((summary for summary in cap_summaries if summary['hypothetical_cap'] == 9), None)
    if rank_9 and rank_9.get('selected_for_marking'):
        if any(report['candidate_review_class'] == 'weak_geometry_rank_below_cap' for report in review_reports):
            return 'do_not_increase_cap_due_to_weak_candidates'
        return 'keep_cap_8'
    if rank_9 and rank_9['candidate_review_class'] == 'strong_human_rank_below_cap':
        return 'visually_review_rank_9_only'
    if cap_9 and cap_9['risk_summary'] == 'low':
        return 'consider_cap_9_after_visual_confirmation'
    if any(report['candidate_review_class'] == 'weak_geometry_rank_below_cap' for report in review_reports):
        return 'do_not_increase_cap_due_to_weak_candidates'
    if any(report['candidate_review_class'] == 'strong_human_rank_below_cap' for report in review_reports[:4]):
        return 'visually_review_rank_9_to_12'
    return 'keep_cap_8'


def _rank_review_special_path_report(review_reports, path):
    canonical = _canonical_two_edge_path(path)
    report = next(
        (item for item in review_reports if _canonical_two_edge_path(item['path_vertex_ids']) == canonical),
        None,
    )
    if report is None:
        return {
            'found_in_review': False,
            'path_vertex_ids': list(path),
            'rank_v2': None,
            'continuity_tier': None,
            'q_floor': None,
            'q_sum': None,
            'rank_delta_from_cap': None,
            'is_highest_ranked_unselected_human_candidate': False,
            'would_be_selected_if_cap_9': False,
            'duplicate_endpoint_pair_suppressed': None,
            'visual_review_priority': None,
        }
    human_unselected = [
        item for item in review_reports
        if item['is_residual_human_path'] and not item['selected_for_marking']
    ]
    best_human_rank = min((item['rank_v2'] for item in human_unselected), default=None)
    result = dict(report)
    result.update({
        'found_in_review': True,
        'is_highest_ranked_unselected_human_candidate': bool(
            best_human_rank is not None and report['rank_v2'] == best_human_rank
        ),
    })
    return result


def _rank_review_path_key(report):
    return tuple(report.get('path_vertex_ids', ()))


def _rank_review_candidate_label(report):
    labels = report.get('human_gap_match_labels') or []
    if labels:
        return labels[0]
    target = report.get('old_validation_target_match_label')
    if target:
        return target
    return 'non_human'


def _rank_review_report_is_weak(report):
    return bool(
        (report.get('continuity_tier') is not None and report['continuity_tier'] >= 3)
        or (report.get('path_straightness') is not None and report['path_straightness'] < 0.5)
        or (report.get('q_floor') is not None and report['q_floor'] < 0.5)
    )


def _rank_review_report_is_strong(report):
    return bool(
        report.get('continuity_tier') is not None
        and report['continuity_tier'] <= 1
        and report.get('q_floor') is not None
        and report['q_floor'] >= 0.7
        and report.get('q_sum') is not None
        and report['q_sum'] >= 1.4
        and report.get('path_straightness') is not None
        and report['path_straightness'] >= 0.7
    )


def _min_report_value(reports, key):
    values = [report.get(key) for report in reports if report.get(key) is not None]
    if not values:
        return None
    return min(values)


def _rank_review_cap_added_count(payload, cap):
    for summary in payload.get('hypothetical_cap_summaries', []):
        if summary.get('hypothetical_cap') == cap:
            return summary.get('additional_candidates_selected', 0)
    return 0


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
        'raw_allowed_total': result.blender_two_edge_endpoint_bridge_raw_allowed_total,
        'deduplicated_allowed_total': result.blender_two_edge_endpoint_bridge_deduplicated_allowed_total,
        'allowed_total': result.blender_two_edge_endpoint_bridge_allowed_total,
        'selected_total': result.blender_two_edge_endpoint_bridge_paths_marked,
        'over_cap': result.blender_two_edge_endpoint_bridge_over_cap,
        'selected_rank_threshold': threshold,
        'duplicate_endpoint_pairs_suppressed': (
            result.blender_two_edge_endpoint_bridge_duplicate_endpoint_pairs_suppressed
        ),
        'score_tuple_definition_v1_length_first': list(
            ENDPOINT_BRIDGE_SCORE_TUPLE_DEFINITION_V1_LENGTH_FIRST
        ),
        'score_tuple_definition_v2': list(ENDPOINT_BRIDGE_SCORE_TUPLE_DEFINITION_V2),
        'score_tuple_definition': list(ENDPOINT_BRIDGE_SCORE_TUPLE_DEFINITION_V2),
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
        'added_candidate_due_to_cap_increase': (
            result.blender_two_edge_endpoint_bridge_added_candidate_due_to_cap_increase
        ),
        'previous_rank_9_selected': result.blender_two_edge_endpoint_bridge_previous_rank_9_selected,
        'selected_rank_9_candidate': result.blender_two_edge_endpoint_bridge_selected_rank_9_candidate,
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
    for _, path in DIAGNOSTIC_OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS:
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
    for label, group_id, alternative_id, path in DIAGNOSTIC_HUMAN_GAP_REGRESSION_PATHS:
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
            'rank': report.get('rank_v2'),
            'rank_v1_length_first': report.get('rank_v1_length_first'),
            'rank_v2': report.get('rank_v2'),
            'rank_delta_v2_minus_v1': report.get('rank_delta_v2_minus_v1'),
            'continuity_tier': report.get('continuity_tier'),
            'q_floor': report.get('q_floor'),
            'q_sum': report.get('q_sum'),
            'selected_for_marking': bool(report.get('selected_for_marking', False)),
            'marked': bool(report.get('marked', False)),
            'skipped_reason': report.get('skipped_reason'),
            'candidate_score_tuple': report.get('candidate_score_tuple_v2'),
            'candidate_score_tuple_v2': report.get('candidate_score_tuple_v2'),
            'total_path_length': report.get('total_path_length'),
            'endpoint_distance': report.get('endpoint_distance'),
            'min_endpoint_tangent_alignment': report.get('min_endpoint_tangent_alignment'),
            'path_straightness': report.get('path_straightness'),
            'rank_delta_from_threshold': _rank_delta(report.get('rank_v2'), threshold),
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
    for label, path in DIAGNOSTIC_OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS:
        target_label = _old_validation_target_label(path)
        report = by_path.get(_canonical_two_edge_path(path))
        if report is None:
            reports[target_label] = {
                'found_in_allowed_candidates': False,
                'rank': None,
                'rank_v1_length_first': None,
                'rank_v2': None,
                'rank_delta_v2_minus_v1': None,
                'selected_for_marking': False,
                'marked': False,
                'skipped_reason': 'not_found',
                'candidate_score_tuple': None,
                'candidate_score_tuple_v1_length_first': None,
                'candidate_score_tuple_v2': None,
                'continuity_tier': None,
                'q_floor': None,
                'q_sum': None,
                'total_path_length': None,
                'endpoint_distance': None,
                'endpoint_tangent_alignment_u': None,
                'endpoint_tangent_alignment_v': None,
                'min_endpoint_tangent_alignment': None,
                'path_straightness': None,
                'rank_delta_from_threshold': None,
                'primary_penalty_component': 'not_found',
                'selected_by_v2_continuity_ranking': False,
            }
            continue
        reports[target_label] = {
            'found_in_allowed_candidates': True,
            'rank': report.get('rank_v2'),
            'rank_v1_length_first': report.get('rank_v1_length_first'),
            'rank_v2': report.get('rank_v2'),
            'rank_delta_v2_minus_v1': report.get('rank_delta_v2_minus_v1'),
            'selected_for_marking': bool(report.get('selected_for_marking', False)),
            'marked': bool(report.get('marked', False)),
            'skipped_reason': report.get('skipped_reason'),
            'candidate_score_tuple': report.get('candidate_score_tuple_v2'),
            'candidate_score_tuple_v1_length_first': report.get(
                'candidate_score_tuple_v1_length_first'
            ),
            'candidate_score_tuple_v2': report.get('candidate_score_tuple_v2'),
            'continuity_tier': report.get('continuity_tier'),
            'q_floor': report.get('q_floor'),
            'q_sum': report.get('q_sum'),
            'total_path_length': report.get('total_path_length'),
            'endpoint_distance': report.get('endpoint_distance'),
            'endpoint_tangent_alignment_u': report.get('endpoint_tangent_alignment_u'),
            'endpoint_tangent_alignment_v': report.get('endpoint_tangent_alignment_v'),
            'min_endpoint_tangent_alignment': report.get('min_endpoint_tangent_alignment'),
            'path_straightness': report.get('path_straightness'),
            'rank_delta_from_threshold': _rank_delta(report.get('rank_v2'), threshold),
            'primary_penalty_component': _primary_penalty_component(
                report,
                threshold_report,
                selected,
            ),
            'selected_by_v2_continuity_ranking': bool(report.get('selected_for_marking', False)),
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
    for label, target_path in DIAGNOSTIC_OLD_ENDPOINT_BRIDGE_VALIDATION_TARGETS:
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
        'continuity_tier',
        'quality_floor',
        'quality_sum',
        'total_path_length',
        'endpoint_distance',
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
    if result.human_case_2557_2558_marked_seam:
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
        f'raw_allowed={result.blender_two_edge_endpoint_bridge_raw_allowed_total}, '
        f'dedup_allowed={result.blender_two_edge_endpoint_bridge_deduplicated_allowed_total}, '
        f'over_cap={str(result.blender_two_edge_endpoint_bridge_over_cap).lower()}, '
        f'policy={result.blender_two_edge_endpoint_bridge_selection_policy}. '
        f'Curved two-edge endpoint bridge: '
        f'{result.blender_curved_two_edge_endpoint_bridge_paths_marked} paths marked, '
        f'{result.blender_curved_two_edge_endpoint_bridge_edges_marked} edges marked, '
        f'eligible={result.blender_curved_two_edge_endpoint_bridge_eligible_total}, '
        f'over_cap={str(result.blender_curved_two_edge_endpoint_bridge_over_cap).lower()}, '
        f'cap={result.blender_curved_two_edge_endpoint_bridge_safety_cap}. '
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
