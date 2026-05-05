import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.utils.filename_parsing import (
    DEFAULT_RESOLUTION_PATTERNS,
    FilenameParseConfig,
    parse_mesh_name,
)
from preprocessing.compute_features import build_edge_topology, detect_symmetry_axis
from preprocessing.build_gnn_dataset import _detect_seam_edges

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None


CSV_FIELDS = [
    'file_path',
    'stem',
    'family_id',
    'resolution_tag',
    'is_augmented',
    'num_vertices',
    'num_faces',
    'num_unique_edges',
    'num_boundary_edges',
    'num_seam_edges',
    'seam_ratio',
    'boundary_seam_ratio',
    'merge_before_vertices',
    'merge_after_vertices',
    'merge_reduction',
    'detected_symmetry_axis',
]

AXIS_NAMES = {0: 'x', 1: 'y', 2: 'z', None: None}


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _json_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _edge_counts_from_faces(faces: np.ndarray) -> tuple[dict[tuple[int, int], int], int]:
    edge_counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for i in range(3):
            a = int(face[i])
            b = int(face[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    return edge_counts, boundary_edges


def _base_row(file_path: str | Path, config: FilenameParseConfig) -> dict[str, Any]:
    info = parse_mesh_name(file_path, config)
    row = asdict(info)
    row['file_path'] = str(file_path)
    return row


def _audit_obj(obj_path: Path, config: FilenameParseConfig) -> dict[str, Any]:
    if trimesh is None:
        raise RuntimeError('trimesh is required to audit raw OBJ files')

    mesh = trimesh.load(str(obj_path), process=False, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError('not a single Trimesh object')
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError('empty mesh')

    seam_map_split = _detect_seam_edges(mesh)
    split_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    merge_before = len(split_vertices)

    mesh.merge_vertices()
    merge_after = len(mesh.vertices)

    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(mesh.vertices, dtype=np.float64))
    _, old_to_new = tree.query(split_vertices)

    seam_edges: set[tuple[int, int]] = set()
    for (vi, vj), is_seam in seam_map_split.items():
        if not is_seam:
            continue
        geo_vi = int(old_to_new[vi])
        geo_vj = int(old_to_new[vj])
        if geo_vi == geo_vj:
            continue
        seam_edges.add((min(geo_vi, geo_vj), max(geo_vi, geo_vj)))

    unique_edges, edge_to_faces = build_edge_topology(mesh)
    boundary_edges = sum(1 for faces in edge_to_faces.values() if len(faces) == 1)
    num_unique_edges = len(unique_edges)
    num_seam_edges = len(seam_edges)
    axis = detect_symmetry_axis(mesh)

    row = _base_row(obj_path, config)
    row.update({
        'num_vertices': int(len(mesh.vertices)),
        'num_faces': int(len(mesh.faces)),
        'num_unique_edges': int(num_unique_edges),
        'num_boundary_edges': int(boundary_edges),
        'num_seam_edges': int(num_seam_edges),
        'seam_ratio': _ratio(num_seam_edges, num_unique_edges),
        'boundary_seam_ratio': _ratio(boundary_edges, num_seam_edges),
        'merge_before_vertices': int(merge_before),
        'merge_after_vertices': int(merge_after),
        'merge_reduction': int(merge_before - merge_after),
        'detected_symmetry_axis': AXIS_NAMES[axis],
    })
    return row


def _tensor_len(value: Any) -> int | None:
    if value is None:
        return None
    return int(value.shape[0]) if hasattr(value, 'shape') else len(value)


def _audit_data(data: Any, fallback_name: str, config: FilenameParseConfig) -> dict[str, Any]:
    file_path = getattr(data, 'file_path', fallback_name)
    row = _base_row(file_path, config)

    faces = getattr(data, 'faces', None)
    num_faces = _tensor_len(faces)
    num_vertices = int(getattr(data, 'num_nodes', 0) or 0) if hasattr(data, 'num_nodes') else None

    edge_index = getattr(data, 'edge_index', None)
    y = getattr(data, 'y', None)
    edge_count = int(edge_index.shape[1]) if edge_index is not None else None
    label_count = int(y.shape[0]) if y is not None and hasattr(y, 'shape') else None

    is_original_graph = edge_count is not None and label_count == edge_count and edge_count % 2 == 0
    if is_original_graph:
        num_unique_edges = edge_count // 2
        labels = y[:num_unique_edges]
    else:
        num_unique_edges = int(getattr(data, 'num_nodes', 0) or label_count or 0)
        labels = y

    num_seam_edges = int(labels.bool().sum().item()) if labels is not None else None

    num_boundary_edges = None
    if faces is not None:
        faces_np = faces.detach().cpu().numpy() if hasattr(faces, 'detach') else np.asarray(faces)
        _, num_boundary_edges = _edge_counts_from_faces(faces_np)

    row.update({
        'num_vertices': num_vertices if is_original_graph else None,
        'num_faces': num_faces,
        'num_unique_edges': num_unique_edges,
        'num_boundary_edges': num_boundary_edges,
        'num_seam_edges': num_seam_edges,
        'seam_ratio': _ratio(num_seam_edges, num_unique_edges),
        'boundary_seam_ratio': _ratio(num_boundary_edges, num_seam_edges),
        'merge_before_vertices': getattr(data, 'merge_before_vertices', None),
        'merge_after_vertices': getattr(data, 'merge_after_vertices', None),
        'merge_reduction': getattr(data, 'merge_reduction', None),
        'detected_symmetry_axis': getattr(data, 'detected_symmetry_axis', None),
    })
    return row


def _load_pt(path: Path) -> list[Any]:
    dataset = torch.load(path, weights_only=False)
    if not isinstance(dataset, list):
        raise ValueError(f'expected a list of Data objects, got {type(dataset)}')
    return dataset


def _split_rows(
    rows: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    import random

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = row['family_id']
        grouped[key].append(idx)

    keys = list(grouped.keys())
    random.Random(seed).shuffle(keys)

    n_test = max(1, int(len(keys) * test_ratio)) if keys else 0
    n_val = max(1, int(len(keys) * val_ratio)) if keys else 0
    split_keys = {
        'test': keys[:n_test],
        'val': keys[n_test:n_test + n_val],
        'train': keys[n_test + n_val:],
    }
    return {name: [idx for key in split_keys[name] for idx in grouped[key]] for name in ('train', 'val', 'test')}


def _leakage_report(rows: list[dict[str, Any]], splits: dict[str, list[int]]) -> dict[str, Any]:
    family_to_splits: dict[str, set[str]] = defaultdict(set)
    family_res_splits: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for split_name, indices in splits.items():
        for idx in indices:
            row = rows[idx]
            family = row['family_id']
            resolution = row['resolution_tag'] or 'unknown'
            family_to_splits[family].add(split_name)
            family_res_splits[family][resolution].add(split_name)

    family_split_leaks = {
        family: sorted(split_names)
        for family, split_names in family_to_splits.items()
        if len(split_names) > 1
    }
    multi_resolution_split_leaks = {}
    for family, by_resolution in family_res_splits.items():
        split_names = set().union(*by_resolution.values()) if by_resolution else set()
        if len(by_resolution) > 1 and len(split_names) > 1:
            multi_resolution_split_leaks[family] = {
                resolution: sorted(names)
                for resolution, names in sorted(by_resolution.items())
            }

    return {
        'family_split_leaks': family_split_leaks,
        'multi_resolution_split_leaks': multi_resolution_split_leaks,
    }


def _summary(rows: list[dict[str, Any]], leakage: dict[str, Any], source: Path) -> dict[str, Any]:
    total_edges = sum(row['num_unique_edges'] or 0 for row in rows)
    total_seams = sum(row['num_seam_edges'] or 0 for row in rows)
    total_boundary = sum(row['num_boundary_edges'] or 0 for row in rows)
    families = {row['family_id'] for row in rows}
    resolutions = {row['resolution_tag'] for row in rows if row['resolution_tag']}

    return {
        'source': str(source),
        'mesh_count': len(rows),
        'family_count': len(families),
        'resolution_tags': sorted(resolutions),
        'augmented_count': sum(1 for row in rows if row['is_augmented']),
        'total_unique_edges': total_edges,
        'total_seam_edges': total_seams,
        'total_boundary_edges': total_boundary,
        'seam_ratio': _ratio(total_seams, total_edges),
        'boundary_seam_ratio': _ratio(total_boundary, total_seams),
        'family_split_leak_count': len(leakage['family_split_leaks']),
        'multi_resolution_split_leak_count': len(leakage['multi_resolution_split_leaks']),
    }


def _print_summary(summary: dict[str, Any], leakage: dict[str, Any], outputs: tuple[Path, Path]) -> None:
    print('Dataset audit')
    print(f"  source: {summary['source']}")
    print(f"  meshes: {summary['mesh_count']}  families: {summary['family_count']}")
    print(f"  augmented: {summary['augmented_count']}")
    print(f"  unique edges: {summary['total_unique_edges']}  seams: {summary['total_seam_edges']}")
    if summary['seam_ratio'] is not None:
        print(f"  seam ratio: {summary['seam_ratio']:.4f}")
    print(f"  split family leaks: {summary['family_split_leak_count']}")
    print(f"  split multi-resolution leaks: {summary['multi_resolution_split_leak_count']}")
    print(f"  json: {outputs[0]}")
    print(f"  csv: {outputs[1]}")

    if leakage['family_split_leaks']:
        examples = list(leakage['family_split_leaks'].items())[:5]
        print('  family leak examples:')
        for family, split_names in examples:
            print(f"    {family}: {', '.join(split_names)}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_value(row.get(field)) for field in CSV_FIELDS})


def _write_json(path: Path, report: dict[str, Any]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, default=_json_value)
        handle.write('\n')


def audit(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input)
    config = FilenameParseConfig(
        augmentation_pattern=args.augmentation_pattern,
        resolution_patterns=tuple(args.resolution_pattern or DEFAULT_RESOLUTION_PATTERNS),
    )

    if source.is_dir():
        obj_paths = sorted(source.glob('**/*.obj'))
        rows = []
        for obj_path in obj_paths:
            try:
                rows.append(_audit_obj(obj_path, config))
            except Exception as exc:
                print(f'[skip] {obj_path}: {exc}')
    elif source.is_file() and source.suffix == '.pt':
        rows = [_audit_data(data, f'graph_{idx}', config) for idx, data in enumerate(_load_pt(source))]
    else:
        raise ValueError(f'expected an OBJ directory or .pt dataset: {source}')

    splits = _split_rows(rows, args.val_ratio, args.test_ratio, args.seed)
    leakage = _leakage_report(rows, splits)
    summary = _summary(rows, leakage, source)
    report = {
        'summary': summary,
        'split': {
            'seed': args.seed,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'sizes': {name: len(indices) for name, indices in splits.items()},
        },
        'filename_parse_config': asdict(config),
        'leakage': leakage,
        'meshes': rows,
    }

    json_path = Path(args.json_out)
    csv_path = Path(args.csv_out)
    _write_json(json_path, report)
    _write_csv(csv_path, rows)
    _print_summary(summary, leakage, (json_path, csv_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit raw OBJ meshes or a serialized PyG dataset.')
    parser.add_argument('input', help='OBJ directory, dataset.pt, or dataset_dual.pt')
    parser.add_argument('--json-out', default='dataset_audit.json', help='JSON report path')
    parser.add_argument('--csv-out', default='dataset_audit.csv', help='CSV report path')
    parser.add_argument('--seed', type=int, default=42, help='Split seed used for leakage simulation')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.10, help='Test split ratio')
    parser.add_argument('--augmentation-pattern', default=r'_aug\d+$')
    parser.add_argument(
        '--resolution-pattern',
        action='append',
        default=None,
        help='Regex suffix to strip for family parsing; can be passed multiple times',
    )
    args = parser.parse_args()
    audit(args)


if __name__ == '__main__':
    main()
