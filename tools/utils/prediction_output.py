from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from preprocessing.feature_registry import ResolvedFeatureSet
from preprocessing.topology import CanonicalTopology
from tools.utils.json_io import write_json


def build_output_payload(
    *,
    mesh_path: Path,
    output_json: Path,
    weights_path: Path,
    config_path: Path,
    model_type: str,
    feature_bundle: str,
    selection: ResolvedFeatureSet,
    threshold: float,
    device: torch.device,
    topology: CanonicalTopology,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    seam_mask: np.ndarray,
    write_all_edges: bool,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seam_edge_indices = [int(idx) for idx in np.flatnonzero(seam_mask)]
    edge_rows = []
    seam_rows = []
    boundary_count = 0

    for idx, edge in enumerate(unique_edges):
        vi, vj = int(edge[0]), int(edge[1])
        edge_key = (vi, vj)
        is_boundary = len(topology.edge_incidence[edge_key]) == 1
        if is_boundary:
            boundary_count += 1
        row = {
            'canonical_edge_index': int(idx),
            'vertex_ids_0based': [vi, vj],
            'vertex_ids_obj_1based': [vi + 1, vj + 1],
            'probability': float(probabilities[idx]),
            'predicted_seam': bool(seam_mask[idx]),
            'is_boundary': bool(is_boundary),
        }
        if write_all_edges:
            edge_rows.append(row)
        if seam_mask[idx]:
            seam_rows.append({
                'canonical_edge_index': row['canonical_edge_index'],
                'vertex_ids_0based': row['vertex_ids_0based'],
                'vertex_ids_obj_1based': row['vertex_ids_obj_1based'],
                'probability': row['probability'],
                'is_boundary': row['is_boundary'],
            })

    payload: dict[str, Any] = {
        'schema_version': 1,
        'status': 'ok',
        'mesh_path': str(mesh_path.resolve()),
        'output_json': str(output_json.resolve()),
        'model': {
            'model_type': model_type,
            'weights_path': str(weights_path.resolve()),
            'config_path': str(config_path.resolve()),
            'feature_bundle': feature_bundle,
            'feature_group': selection.feature_group,
            'feature_names': list(selection.feature_names),
            'in_dim': int(selection.feature_count),
            'threshold': float(threshold),
            'device': str(device),
        },
        'topology': {
            'vertex_count': int(len(topology.canonical_vertices)),
            'face_count': int(len(topology.canonical_faces)),
            'edge_count': int(len(unique_edges)),
            'edge_order': 'canonical_topology_sorted_edge_keys',
        },
        'stats': {
            'predicted_seam_count': int(len(seam_edge_indices)),
            'boundary_edge_count': int(boundary_count),
            'probability_min': float(np.min(probabilities)) if len(probabilities) else 0.0,
            'probability_max': float(np.max(probabilities)) if len(probabilities) else 0.0,
            'probability_mean': float(np.mean(probabilities)) if len(probabilities) else 0.0,
        },
        'seam_edge_indices': seam_edge_indices,
        'seam_edges': seam_rows,
    }
    if diagnostics is not None:
        annotate_bridge_output_presence(
            diagnostics,
            seam_edge_indices=seam_edge_indices,
            seam_rows=seam_rows,
        )
        payload['diagnostics'] = diagnostics
    if write_all_edges:
        payload['edges'] = edge_rows
    return payload


def annotate_bridge_output_presence(
    diagnostics: dict[str, Any],
    *,
    seam_edge_indices: list[int],
    seam_rows: list[dict[str, Any]],
) -> None:
    postprocess = diagnostics.get('postprocess')
    if not isinstance(postprocess, dict):
        return
    bridging = postprocess.get('bridging')
    if not isinstance(bridging, dict):
        return

    output_indices = {int(index) for index in seam_edge_indices}
    output_edges = {int(row['canonical_edge_index']) for row in seam_rows if 'canonical_edge_index' in row}
    accepted_indices = [
        int(index)
        for index in bridging.get('accepted_bridge_edge_indices', [])
        if isinstance(index, int)
    ]
    survived = [index for index in accepted_indices if index in output_indices]

    bridging['accepted_bridge_edges_survived_to_final'] = len(survived)
    bridging['accepted_bridge_edges_removed_by_stage_c'] = len(accepted_indices) - len(survived)
    bridging['final_output_contains_accepted_bridge_edges'] = bool(survived)

    existing_reports = bridging.get('bridge_edge_ids_final_presence')
    if not isinstance(existing_reports, list):
        existing_reports = []
    report_by_id = {
        int(report['edge_id']): dict(report)
        for report in existing_reports
        if isinstance(report, dict) and isinstance(report.get('edge_id'), int)
    }
    for edge_id in accepted_indices:
        report = report_by_id.setdefault(edge_id, {'edge_id': edge_id})
        report['in_output_seam_edge_indices'] = edge_id in output_indices
        report['in_output_seam_edges'] = edge_id in output_edges
        report.setdefault('original_blender_edge_if_traceable', None)
        report.setdefault('applied_by_blender_if_traceable', None)
    bridging['bridge_edge_ids_final_presence'] = [
        report_by_id[edge_id]
        for edge_id in sorted(report_by_id)
    ]


def write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def write_error_payload(path: Path, error_type: str, message: str) -> None:
    write_json(path, {
        'schema_version': 1,
        'status': 'error',
        'error_type': error_type,
        'message': message,
    })
