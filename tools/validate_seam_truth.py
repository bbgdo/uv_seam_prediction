import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.obj_parser import ObjCorner, ObjMesh, parse_obj
from preprocessing.seam_labels import SeamTruth, extract_seam_truth
from preprocessing.topology import (
    CanonicalTopology,
    EdgeKey,
    WeldConfig,
    build_topology,
    canonical_edge_key,
)


OccurrenceId = tuple[int, int]
UVSignature = tuple[int | None, int | None]


def _iter_meshes(args: argparse.Namespace) -> list[Path]:
    if args.mesh:
        paths = [Path(path) for path in args.mesh]
    else:
        paths = sorted(Path(args.mesh_dir).glob('**/*.obj'))
    if args.max_meshes is not None:
        paths = paths[:args.max_meshes]
    if not paths:
        raise ValueError('no OBJ files found')
    return paths


def _align_signature_direct(
    edge_key: EdgeKey,
    gid_a: int,
    gid_b: int,
    corner_a: ObjCorner,
    corner_b: ObjCorner,
) -> UVSignature:
    if (gid_a, gid_b) == edge_key:
        return corner_a.uv_index, corner_b.uv_index
    if (gid_b, gid_a) == edge_key:
        return corner_b.uv_index, corner_a.uv_index
    raise ValueError(f'corner gids {(gid_a, gid_b)} do not match edge {edge_key}')


def direct_reference_from_obj(mesh: ObjMesh, topology: CanonicalTopology) -> tuple[
    dict[EdgeKey, bool],
    dict[EdgeKey, bool],
    dict[OccurrenceId, UVSignature],
]:
    edge_occurrences: dict[EdgeKey, list[tuple[OccurrenceId, UVSignature]]] = defaultdict(list)

    for face_index, face in enumerate(mesh.faces):
        for local_edge_index, (a_idx, b_idx) in enumerate(((0, 1), (1, 2), (2, 0))):
            corner_a = face.corners[a_idx]
            corner_b = face.corners[b_idx]
            gid_a = topology.original_vertex_to_canonical_gid[corner_a.vertex_index]
            gid_b = topology.original_vertex_to_canonical_gid[corner_b.vertex_index]
            edge_key = canonical_edge_key(gid_a, gid_b)
            signature = _align_signature_direct(edge_key, gid_a, gid_b, corner_a, corner_b)
            edge_occurrences[edge_key].append(((face_index, local_edge_index), signature))

    seam_map = {}
    boundary_map = {}
    signatures_by_occurrence = {}
    for edge_key, occurrences in edge_occurrences.items():
        for occurrence_id, signature in occurrences:
            signatures_by_occurrence[occurrence_id] = signature

        if len(occurrences) == 1:
            boundary_map[edge_key] = True
            seam_map[edge_key] = True
        elif len(occurrences) == 2:
            boundary_map[edge_key] = False
            seam_map[edge_key] = occurrences[0][1] != occurrences[1][1]
        else:
            raise ValueError(f'non-manifold edge {edge_key}: {len(occurrences)} incident faces')

    return seam_map, boundary_map, signatures_by_occurrence


def compare_maps(pipeline: dict[EdgeKey, bool], reference: dict[EdgeKey, bool]) -> dict[str, Any]:
    pipeline_seams = {edge for edge, is_seam in pipeline.items() if is_seam}
    reference_seams = {edge for edge, is_seam in reference.items() if is_seam}
    tp = len(pipeline_seams & reference_seams)
    fp_edges = sorted(pipeline_seams - reference_seams)
    fn_edges = sorted(reference_seams - pipeline_seams)
    return {
        'tp': tp,
        'fp': len(fp_edges),
        'fn': len(fn_edges),
        'fp_edges': fp_edges,
        'fn_edges': fn_edges,
        'mismatch_count': len(fp_edges) + len(fn_edges),
        'seam_count': len(pipeline_seams),
    }


def _corner_to_dict(corner: ObjCorner) -> dict[str, int | None]:
    return {
        'vertex_index': corner.vertex_index,
        'uv_index': corner.uv_index,
        'normal_index': corner.normal_index,
    }


def _edge_debug(
    edge_key: EdgeKey,
    truth: SeamTruth,
    reference_signatures: dict[OccurrenceId, UVSignature],
) -> dict[str, Any]:
    occurrences = truth.incident_face_provenance.get(edge_key, ())
    return {
        'edge_key': list(edge_key),
        'pipeline_is_seam': truth.seam_map.get(edge_key),
        'reference_signatures': [
            list(reference_signatures[(occ.face_index, occ.local_edge_index)])
            for occ in occurrences
        ],
        'pipeline_signatures': [
            list(truth.uv_signature_by_occurrence[(occ.face_index, occ.local_edge_index)])
            for occ in occurrences
        ],
        'occurrences': [
            {
                'face_index': occ.face_index,
                'face_line_number': occ.face_line_number,
                'local_edge_index': occ.local_edge_index,
                'corner_a': _corner_to_dict(occ.corner_a),
                'corner_b': _corner_to_dict(occ.corner_b),
            }
            for occ in occurrences
        ],
    }


def _write_mismatch_debug(
    debug_dir: Path | None,
    obj_path: Path,
    comparison: dict[str, Any],
    truth: SeamTruth,
    reference_signatures: dict[OccurrenceId, UVSignature],
) -> None:
    if debug_dir is None or comparison['mismatch_count'] == 0:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    mismatch_edges = comparison['fp_edges'] + comparison['fn_edges']
    payload = {
        'mesh': str(obj_path),
        'fp_edges': [list(edge) for edge in comparison['fp_edges']],
        'fn_edges': [list(edge) for edge in comparison['fn_edges']],
        'edges': [
            _edge_debug(edge, truth, reference_signatures)
            for edge in mismatch_edges
        ],
    }
    out_path = debug_dir / f'{obj_path.stem}_seam_truth_mismatch.json'
    with out_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')


def validate_one(obj_path: Path, weld_config: WeldConfig, debug_dir: Path | None) -> dict[str, Any]:
    mesh = parse_obj(obj_path)
    topology = build_topology(mesh, weld_config)
    truth = extract_seam_truth(topology)
    reference_seams, reference_boundaries, reference_signatures = direct_reference_from_obj(mesh, topology)
    comparison = compare_maps(truth.seam_map, reference_seams)

    boundary_count = sum(1 for is_boundary in truth.boundary_map.values() if is_boundary)
    reference_boundary_count = sum(1 for is_boundary in reference_boundaries.values() if is_boundary)
    if boundary_count != reference_boundary_count:
        comparison['mismatch_count'] += 1

    _write_mismatch_debug(debug_dir, obj_path, comparison, truth, reference_signatures)

    return {
        'mesh': str(obj_path),
        'tp': comparison['tp'],
        'fp': comparison['fp'],
        'fn': comparison['fn'],
        'mismatch_count': comparison['mismatch_count'],
        'seam_count': comparison['seam_count'],
        'boundary_edge_count': boundary_count,
        'non_manifold_count': truth.audit.non_manifold_edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Parity-check exact OBJ seam truth extraction.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--mesh', action='append', help='OBJ path; can be passed multiple times')
    source.add_argument('--mesh-dir', help='Directory of OBJ files')
    parser.add_argument('--max-meshes', type=int)
    parser.add_argument('--weld-quantization', type=float, default=None)
    parser.add_argument('--debug-dir', type=Path, default=None)
    parser.add_argument('--allow-mismatch', action='store_true')
    args = parser.parse_args()

    weld_config = (
        WeldConfig.welded(args.weld_quantization)
        if args.weld_quantization is not None
        else WeldConfig.exact()
    )
    results = []
    for obj_path in _iter_meshes(args):
        result = validate_one(obj_path, weld_config, args.debug_dir)
        results.append(result)
        print(
            f"{Path(result['mesh']).name}: "
            f"TP={result['tp']} FP={result['fp']} FN={result['fn']} "
            f"mismatches={result['mismatch_count']} seams={result['seam_count']} "
            f"boundary={result['boundary_edge_count']} non_manifold={result['non_manifold_count']}"
        )

    total_mismatches = sum(result['mismatch_count'] for result in results)
    print(f'summary: meshes={len(results)} mismatches={total_mismatches}')
    if total_mismatches and not args.allow_mismatch:
        sys.exit(1)


if __name__ == '__main__':
    main()
