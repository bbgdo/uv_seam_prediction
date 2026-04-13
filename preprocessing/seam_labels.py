import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from preprocessing.obj_parser import ObjCorner, parse_obj
    from preprocessing.topology import (
        CanonicalTopology,
        EdgeKey,
        FaceEdgeOccurrence,
        WeldConfig,
        build_topology,
    )
except ModuleNotFoundError:  # pragma: no cover - supports `python preprocessing/seam_labels.py`
    from obj_parser import ObjCorner, parse_obj
    from topology import (
        CanonicalTopology,
        EdgeKey,
        FaceEdgeOccurrence,
        WeldConfig,
        build_topology,
    )


class SeamLabelError(ValueError):
    pass


UVSignature = tuple[int | None, int | None]
OccurrenceId = tuple[int, int]


@dataclass(frozen=True)
class SeamAudit:
    edge_count: int
    seam_edges: int
    boundary_edges: int
    non_manifold_edges: int
    missing_uv_occurrences: int


@dataclass(frozen=True)
class SeamTruth:
    seam_map: dict[EdgeKey, bool]
    boundary_map: dict[EdgeKey, bool]
    uv_signature_by_occurrence: dict[OccurrenceId, UVSignature]
    incident_face_provenance: dict[EdgeKey, tuple[FaceEdgeOccurrence, ...]]
    audit: SeamAudit


def aligned_uv_signature(
    occurrence: FaceEdgeOccurrence,
    topology: CanonicalTopology,
) -> UVSignature:
    """Return UV ids aligned to the canonical geometric edge endpoint order.

    OBJ UV indices are the seam truth because UV splits are represented by
    different face-corner `vt` ids. Missing UVs use a deterministic sentinel so
    no tolerance or coordinate comparison is hidden in this path.
    """
    gid_a = topology.original_vertex_to_canonical_gid[occurrence.corner_a.vertex_index]
    gid_b = topology.original_vertex_to_canonical_gid[occurrence.corner_b.vertex_index]
    local_key = (gid_a, gid_b)

    if local_key == occurrence.edge_key:
        return occurrence.corner_a.uv_index, occurrence.corner_b.uv_index
    if (gid_b, gid_a) == occurrence.edge_key:
        return occurrence.corner_b.uv_index, occurrence.corner_a.uv_index
    raise SeamLabelError(
        f'face {occurrence.face_index} edge {occurrence.local_edge_index}: '
        f'corner vertices do not match edge {occurrence.edge_key}'
    )


def extract_seam_truth(topology: CanonicalTopology) -> SeamTruth:
    seam_map: dict[EdgeKey, bool] = {}
    boundary_map: dict[EdgeKey, bool] = {}
    uv_signature_by_occurrence: dict[OccurrenceId, UVSignature] = {}
    missing_uv_occurrences = 0
    non_manifold_edges = 0

    for edge_key, occurrences in topology.edge_incidence.items():
        signatures = []
        for occurrence in occurrences:
            signature = aligned_uv_signature(occurrence, topology)
            if signature[0] is None or signature[1] is None:
                missing_uv_occurrences += 1
            uv_signature_by_occurrence[(occurrence.face_index, occurrence.local_edge_index)] = signature
            signatures.append(signature)

        if len(occurrences) == 1:
            boundary_map[edge_key] = True
            seam_map[edge_key] = True
        elif len(occurrences) == 2:
            boundary_map[edge_key] = False
            seam_map[edge_key] = signatures[0] != signatures[1]
        else:
            non_manifold_edges += 1
            raise SeamLabelError(f'non-manifold edge {edge_key}: {len(occurrences)} incident faces')

    audit = SeamAudit(
        edge_count=len(seam_map),
        seam_edges=sum(1 for is_seam in seam_map.values() if is_seam),
        boundary_edges=sum(1 for is_boundary in boundary_map.values() if is_boundary),
        non_manifold_edges=non_manifold_edges,
        missing_uv_occurrences=missing_uv_occurrences,
    )
    return SeamTruth(
        seam_map=seam_map,
        boundary_map=boundary_map,
        uv_signature_by_occurrence=uv_signature_by_occurrence,
        incident_face_provenance=topology.edge_incidence,
        audit=audit,
    )


def _corner_to_dict(corner: ObjCorner) -> dict[str, int | None]:
    return {
        'vertex_index': corner.vertex_index,
        'uv_index': corner.uv_index,
        'normal_index': corner.normal_index,
    }


def seam_truth_to_jsonable(truth: SeamTruth) -> dict[str, Any]:
    return {
        'audit': truth.audit.__dict__,
        'edges': [
            {
                'edge_key': list(edge_key),
                'is_seam': truth.seam_map[edge_key],
                'is_boundary': truth.boundary_map[edge_key],
                'occurrences': [
                    {
                        'face_index': occurrence.face_index,
                        'face_line_number': occurrence.face_line_number,
                        'local_edge_index': occurrence.local_edge_index,
                        'uv_signature': list(
                            truth.uv_signature_by_occurrence[
                                (occurrence.face_index, occurrence.local_edge_index)
                            ]
                        ),
                        'corner_a': _corner_to_dict(occurrence.corner_a),
                        'corner_b': _corner_to_dict(occurrence.corner_b),
                    }
                    for occurrence in truth.incident_face_provenance[edge_key]
                ],
            }
            for edge_key in sorted(truth.seam_map)
        ],
    }


def write_seam_truth_json(truth: SeamTruth, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as handle:
        json.dump(seam_truth_to_jsonable(truth), handle, indent=2)
        handle.write('\n')


def write_seam_edges_txt(truth: SeamTruth, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as handle:
        for edge_key in sorted(edge for edge, is_seam in truth.seam_map.items() if is_seam):
            handle.write(f'{edge_key[0]} {edge_key[1]}\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract exact seam truth from OBJ face-corner topology.')
    parser.add_argument('obj_path')
    parser.add_argument('--weld-quantization', type=float, default=None)
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--txt-out', type=Path)
    args = parser.parse_args()

    mesh = parse_obj(args.obj_path)
    weld_config = (
        WeldConfig.welded(args.weld_quantization)
        if args.weld_quantization is not None
        else WeldConfig.exact()
    )
    topology = build_topology(mesh, weld_config)
    truth = extract_seam_truth(topology)

    print(f'file: {mesh.file_path}')
    print(f'edges: {truth.audit.edge_count}')
    print(f'seams: {truth.audit.seam_edges}')
    print(f'boundary edges: {truth.audit.boundary_edges}')
    print(f'missing uv occurrences: {truth.audit.missing_uv_occurrences}')

    if args.json_out:
        write_seam_truth_json(truth, args.json_out)
    if args.txt_out:
        write_seam_edges_txt(truth, args.txt_out)


if __name__ == '__main__':
    main()
