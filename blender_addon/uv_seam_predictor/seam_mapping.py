import json
from dataclasses import dataclass

import bmesh


@dataclass
class SeamApplyResult:
    requested: int
    unique: int
    applied: int
    missing: int
    duplicates_skipped: int


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


def apply_seam_keys(mesh, predicted_keys, clear_existing=True):
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.verts.index_update()

        edge_by_key = {}
        for edge in bm.edges:
            v0 = edge.verts[0].index
            v1 = edge.verts[1].index
            edge_by_key[(min(v0, v1), max(v0, v1))] = edge

        if clear_existing:
            for edge in bm.edges:
                edge.seam = False

        requested = len(predicted_keys)
        seen = set()
        applied = 0
        missing = 0
        duplicates_skipped = 0

        for key in predicted_keys:
            if key in seen:
                duplicates_skipped += 1
                continue
            seen.add(key)

            edge = edge_by_key.get(key)
            if edge is None:
                missing += 1
                continue

            edge.seam = True
            applied += 1

        bm.to_mesh(mesh)
        mesh.update()

        return SeamApplyResult(
            requested=requested,
            unique=len(seen),
            applied=applied,
            missing=missing,
            duplicates_skipped=duplicates_skipped,
        )
    finally:
        bm.free()


def _is_vertex_pair(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and type(value[0]) is int
        and type(value[1]) is int
        and value[0] >= 0
        and value[1] >= 0
    )
