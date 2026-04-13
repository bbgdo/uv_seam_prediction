import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'preprocessing'))

from evaluation.uv_metrics import parse_obj_with_uv
from preprocessing.obj_to_dataset_graph import _detect_seam_edges

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None


EdgeKey = tuple[int, int]


def _edge_key(a: int, b: int) -> EdgeKey:
    return (a, b) if a < b else (b, a)


def audit_vertex_remap(
    split_vertices: np.ndarray,
    merged_vertices: np.ndarray,
    error_eps: float = 1e-8,
    ambiguous_eps: float = 1e-9,
) -> dict[str, Any]:
    if cKDTree is None:
        raise RuntimeError('scipy is required for remap validation')

    tree = cKDTree(merged_vertices)
    k = 2 if len(merged_vertices) > 1 else 1
    distances, old_to_new = tree.query(split_vertices, k=k)

    if k == 1:
        nearest = np.asarray(old_to_new, dtype=np.int64)
        nearest_dist = np.asarray(distances, dtype=np.float64)
        second_dist = np.full(len(split_vertices), np.inf, dtype=np.float64)
    else:
        nearest = np.asarray(old_to_new[:, 0], dtype=np.int64)
        nearest_dist = np.asarray(distances[:, 0], dtype=np.float64)
        second_dist = np.asarray(distances[:, 1], dtype=np.float64)

    counts = Counter(int(idx) for idx in nearest)
    many_to_one_vertices = sum(1 for count in counts.values() if count > 1)
    many_to_one_split_vertices = sum(count for count in counts.values() if count > 1)

    nonzero = nearest_dist > error_eps
    ambiguous = (second_dist - nearest_dist) <= ambiguous_eps
    ambiguous &= second_dist < np.inf

    max_error = float(nearest_dist.max()) if len(nearest_dist) else 0.0
    suspicious = bool(np.any(nonzero) or np.any(ambiguous))

    return {
        'old_to_new': nearest,
        'many_to_one_vertices': int(many_to_one_vertices),
        'many_to_one_split_vertices': int(many_to_one_split_vertices),
        'ambiguous_matches': int(np.sum(ambiguous)),
        'nonzero_reconstruction_errors': int(np.sum(nonzero)),
        'max_reconstruction_error': max_error,
        'suspicious': suspicious,
    }


def _load_split_and_merged_mesh(obj_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    if trimesh is None:
        raise RuntimeError('trimesh is required')

    split_mesh = trimesh.load(str(obj_path), process=False, force='mesh')
    if not isinstance(split_mesh, trimesh.Trimesh):
        raise ValueError(f'not a single mesh: {obj_path}')
    if len(split_mesh.vertices) == 0 or len(split_mesh.faces) == 0:
        raise ValueError(f'empty mesh: {obj_path}')

    split_vertices = np.asarray(split_mesh.vertices, dtype=np.float64).copy()
    merged_mesh = split_mesh.copy()
    merged_mesh.merge_vertices()
    merged_vertices = np.asarray(merged_mesh.vertices, dtype=np.float64)
    audit = audit_vertex_remap(split_vertices, merged_vertices)
    return split_mesh, merged_mesh, audit


def pipeline_seams_from_obj(obj_path: Path) -> tuple[set[EdgeKey], set[EdgeKey], dict[str, Any]]:
    split_mesh, merged_mesh, audit = _load_split_and_merged_mesh(obj_path)
    old_to_new = audit['old_to_new']
    seam_map_split = _detect_seam_edges(split_mesh)

    all_edges: set[EdgeKey] = set()
    seam_edges: set[EdgeKey] = set()
    for (vi, vj), is_seam in seam_map_split.items():
        geo_vi = int(old_to_new[vi])
        geo_vj = int(old_to_new[vj])
        if geo_vi == geo_vj:
            continue
        key = _edge_key(geo_vi, geo_vj)
        all_edges.add(key)
        if is_seam:
            seam_edges.add(key)

    # The denominator must be merged topology, not only edges that survived remap.
    merged_edges = {
        _edge_key(int(edge[0]), int(edge[1]))
        for edge in merged_mesh.edges_unique
    }
    return seam_edges, merged_edges, audit


def reference_seams_from_obj_uv(obj_path: Path, old_to_new: np.ndarray) -> set[EdgeKey]:
    parsed = parse_obj_with_uv(str(obj_path))
    faces = parsed['faces']
    uv_faces = parsed['uv_faces']

    edge_to_uvs: dict[EdgeKey, list[tuple[int, int]]] = {}
    if uv_faces is None:
        uv_faces = faces

    for face, uv_tri in zip(faces, uv_faces):
        for i in range(3):
            vi = int(old_to_new[int(face[i])])
            vj = int(old_to_new[int(face[(i + 1) % 3])])
            if vi == vj:
                continue
            ui = int(uv_tri[i])
            uj = int(uv_tri[(i + 1) % 3])
            edge_to_uvs.setdefault(_edge_key(vi, vj), []).append((ui, uj))

    seams: set[EdgeKey] = set()
    for key, uv_edges in edge_to_uvs.items():
        if len(uv_edges) != 2:
            seams.add(key)
            continue
        a0, b0 = uv_edges[0]
        a1, b1 = uv_edges[1]
        same = (a0 == a1 and b0 == b1) or (a0 == b1 and b0 == a1)
        if not same:
            seams.add(key)
    return seams


def read_blender_edge_list(path: Path) -> set[EdgeKey]:
    """Read merged-topology edge keys exported as `vi vj` or `vi,vj` per line."""
    edges: set[EdgeKey] = set()
    with path.open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            clean = line.split('#', 1)[0].strip()
            if not clean:
                continue
            parts = clean.replace(',', ' ').split()
            if len(parts) != 2:
                raise ValueError(f'{path}:{line_no}: expected two vertex indices')
            edges.add(_edge_key(int(parts[0]), int(parts[1])))
    return edges


def dataset_pipeline_seams(dataset_path: Path) -> dict[str, tuple[set[EdgeKey], set[EdgeKey]]]:
    dataset = torch.load(dataset_path, weights_only=False)
    if not isinstance(dataset, list):
        raise ValueError(f'expected a list of Data objects: {dataset_path}')

    result = {}
    for idx, data in enumerate(dataset):
        file_path = Path(getattr(data, 'file_path', f'graph_{idx}'))
        edge_index = data.edge_index
        num_unique = edge_index.shape[1] // 2
        edges = {
            _edge_key(int(edge_index[0, i]), int(edge_index[1, i]))
            for i in range(num_unique)
        }
        labels = data.y[:num_unique].bool()
        seams = {
            _edge_key(int(edge_index[0, i]), int(edge_index[1, i]))
            for i in range(num_unique)
            if bool(labels[i])
        }
        for key in {file_path.name, file_path.stem}:
            result[key] = (seams, edges)
    return result


def compare_seams(pipeline: set[EdgeKey], reference: set[EdgeKey], edge_count: int) -> dict[str, Any]:
    tp = len(pipeline & reference)
    fp = len(pipeline - reference)
    fn = len(reference - pipeline)
    mismatch_count = fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'mismatch_count': mismatch_count,
        'mismatch_ratio': mismatch_count / max(edge_count, 1),
        'pipeline_seam_ratio': len(pipeline) / max(edge_count, 1),
        'reference_seam_ratio': len(reference) / max(edge_count, 1),
    }


def _write_edge_keys(path: Path, edges: set[EdgeKey]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for vi, vj in sorted(edges):
            handle.write(f'{vi} {vj}\n')


def _write_diff(path: Path, fp: set[EdgeKey], fn: set[EdgeKey]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['kind', 'vi', 'vj'])
        for vi, vj in sorted(fp):
            writer.writerow(['FP', vi, vj])
        for vi, vj in sorted(fn):
            writer.writerow(['FN', vi, vj])


def _debug_artifacts(
    debug_dir: Path | None,
    stem: str,
    pipeline: set[EdgeKey],
    reference: set[EdgeKey],
) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    _write_edge_keys(debug_dir / f'{stem}_pipeline_edges.txt', pipeline)
    _write_edge_keys(debug_dir / f'{stem}_reference_edges.txt', reference)
    _write_diff(debug_dir / f'{stem}_diff.csv', pipeline - reference, reference - pipeline)


def _reference_list_path(args: argparse.Namespace, obj_path: Path) -> Path:
    if args.reference_list:
        return Path(args.reference_list)
    if args.reference_list_dir:
        return Path(args.reference_list_dir) / f'{obj_path.stem}.txt'
    raise ValueError('--reference-list or --reference-list-dir is required for blender-list mode')


def _iter_meshes(args: argparse.Namespace) -> list[Path]:
    if args.mesh:
        meshes = [Path(p) for p in args.mesh]
    else:
        meshes = sorted(Path(args.mesh_dir).glob('**/*.obj'))
    if args.max_meshes is not None:
        meshes = meshes[:args.max_meshes]
    if not meshes:
        raise ValueError('no meshes found')
    return meshes


def validate_mesh(
    obj_path: Path,
    args: argparse.Namespace,
    dataset_lookup: dict[str, tuple[set[EdgeKey], set[EdgeKey]]] | None,
) -> dict[str, Any]:
    pipeline_from_obj, merged_edges, remap = pipeline_seams_from_obj(obj_path)

    if dataset_lookup is not None:
        pipeline, edge_set = dataset_lookup.get(obj_path.name, dataset_lookup.get(obj_path.stem, (None, None)))
        if pipeline is None:
            raise ValueError(f'{obj_path.name}: not found in dataset')
    else:
        pipeline = pipeline_from_obj
        edge_set = merged_edges

    if args.reference_mode == 'obj-uv':
        reference = reference_seams_from_obj_uv(obj_path, remap['old_to_new'])
    else:
        reference = read_blender_edge_list(_reference_list_path(args, obj_path))

    metrics = compare_seams(pipeline, reference, len(edge_set))
    _debug_artifacts(args.debug_dir, obj_path.stem, pipeline, reference)

    suspicious = remap['suspicious']
    has_mismatch = metrics['mismatch_count'] > 0
    status = 'SUSPICIOUS' if suspicious or has_mismatch else 'ok'

    return {
        'file_path': str(obj_path),
        'status': status,
        'edge_count': len(edge_set),
        'pipeline_seams': len(pipeline),
        'reference_seams': len(reference),
        'metrics': metrics,
        'remap_audit': {k: v for k, v in remap.items() if k != 'old_to_new'},
    }


def _print_result(result: dict[str, Any]) -> None:
    metrics = result['metrics']
    remap = result['remap_audit']
    print(f"{Path(result['file_path']).name}: {result['status']}")
    print(
        f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
    )
    print(
        f"  mismatches={metrics['mismatch_count']} "
        f"ratio={metrics['mismatch_ratio']:.6f} "
        f"pipeline_seams={result['pipeline_seams']} reference_seams={result['reference_seams']}"
    )
    print(
        f"  remap many_to_one={remap['many_to_one_vertices']} "
        f"ambiguous={remap['ambiguous_matches']} "
        f"nonzero_error={remap['nonzero_reconstruction_errors']} "
        f"max_error={remap['max_reconstruction_error']:.3e}"
    )
    if result['status'] == 'SUSPICIOUS':
        print('  [SUSPICIOUS] inspect seam diffs and remap audit')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Validate pipeline seam labels against OBJ UVs or Blender edge-key exports.',
        epilog='Blender reference list format: one merged-topology edge key per line as "vi vj" or "vi,vj".',
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--mesh', action='append', help='OBJ mesh path; can be passed multiple times')
    source.add_argument('--mesh-dir', help='Directory of OBJ meshes')
    parser.add_argument('--max-meshes', type=int, help='Limit meshes from --mesh-dir')
    parser.add_argument('--dataset', help='Optional dataset.pt to validate stored labels')
    parser.add_argument('--reference-mode', choices=['obj-uv', 'blender-list'], default='obj-uv')
    parser.add_argument('--reference-list', help='Merged-topology edge list for one mesh')
    parser.add_argument('--reference-list-dir', help='Directory of <mesh_stem>.txt edge lists')
    parser.add_argument('--debug-dir', type=Path, help='Write edge-key and diff artifacts')
    parser.add_argument('--json-out', type=Path, help='Optional JSON summary path')
    args = parser.parse_args()

    dataset_lookup = dataset_pipeline_seams(Path(args.dataset)) if args.dataset else None
    results = []
    for obj_path in _iter_meshes(args):
        result = validate_mesh(obj_path, args, dataset_lookup)
        results.append(result)
        _print_result(result)

    total = {
        'meshes': len(results),
        'suspicious': sum(1 for result in results if result['status'] == 'SUSPICIOUS'),
        'tp': sum(result['metrics']['tp'] for result in results),
        'fp': sum(result['metrics']['fp'] for result in results),
        'fn': sum(result['metrics']['fn'] for result in results),
    }
    print(
        f"summary: meshes={total['meshes']} suspicious={total['suspicious']} "
        f"TP={total['tp']} FP={total['fp']} FN={total['fn']}"
    )

    if args.json_out:
        payload = {'summary': total, 'results': results}
        with args.json_out.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')


if __name__ == '__main__':
    main()
