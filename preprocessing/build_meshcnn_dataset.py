from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from preprocessing._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from models.meshcnn_full.mesh import SPARSE_MESHCNN_SAMPLE_FORMAT, MeshCNNSample, build_mesh_adjacency  # noqa: E402
from preprocessing.canonical_mesh import build_feature_mesh_from_topology, resolve_endpoint_order  # noqa: E402
from preprocessing.compute_features import ENDPOINT_ORDERS, compute_edge_features_for_selection  # noqa: E402
from preprocessing.feature_registry import FEATURE_GROUP_NAMES, ResolvedFeatureSet, resolve_feature_selection  # noqa: E402
from preprocessing.label_sources import EXACT_OBJ_LABEL_SOURCE  # noqa: E402
from preprocessing.obj_parser import parse_obj  # noqa: E402
from preprocessing.seam_labels import extract_seam_truth  # noqa: E402
from preprocessing.topology import WeldConfig, build_topology  # noqa: E402


DEFAULT_OUTPUT = 'dataset_sparsemeshcnn_paper14.pt'


def manifest_path_for_dataset(dataset_path: Path) -> Path:
    return dataset_path.with_name(f'{dataset_path.stem}_manifest.json')


def _assert_exact_edge_order(unique_edges: np.ndarray, canonical_edges: tuple, file_path: Path) -> None:
    expected = np.asarray(canonical_edges, dtype=np.int64).reshape((-1, 2))
    if np.array_equal(unique_edges, expected):
        return
    detail = ''
    if unique_edges.shape == expected.shape:
        mismatch = np.flatnonzero(np.any(unique_edges != expected, axis=1))
        if len(mismatch):
            idx = int(mismatch[0])
            detail = f'; first mismatch at {idx}: features={tuple(unique_edges[idx])}, topology={tuple(expected[idx])}'
    else:
        detail = f'; features shape={unique_edges.shape}, topology shape={expected.shape}'
    raise ValueError(f'exact edge order mismatch for {file_path.name}{detail}')


def _feature_names_for_edge_features(
    feature_selection: ResolvedFeatureSet,
    edge_features: np.ndarray,
    file_path: Path,
) -> list[str]:
    if edge_features.ndim != 2:
        raise ValueError(f'{file_path.name}: edge_features must be rank-2, got shape {edge_features.shape}')

    feature_names = list(feature_selection.feature_names)
    feature_dim = int(edge_features.shape[1])
    if len(feature_names) != feature_dim:
        raise ValueError(
            f'{file_path.name}: resolved feature_names length {len(feature_names)} does not match '
            f'computed edge feature dim {feature_dim}; feature_names={feature_names}'
        )
    return feature_names


def validate_saved_meshcnn_feature_metadata(samples: list[MeshCNNSample]) -> None:
    for sample_idx, sample in enumerate(samples):
        feature_names = list(getattr(sample, 'feature_names', []))
        edge_features = getattr(sample, 'edge_features', None)
        if edge_features is None or getattr(edge_features, 'ndim', 0) != 2:
            raise ValueError(f'MeshCNN sample {sample_idx} is missing rank-2 edge_features tensor')

        feature_dim = int(edge_features.shape[1])
        if len(feature_names) != feature_dim:
            raise ValueError(
                f'MeshCNN sample {sample_idx} feature_names length {len(feature_names)} '
                f'does not match edge_features dim {feature_dim}; '
                f'file_path={getattr(sample, "file_path", "<unknown>")}, '
                f'feature_names={feature_names}'
            )


def build_meshcnn_sample(
    obj_path: str | Path,
    feature_selection: ResolvedFeatureSet,
    endpoint_order: str = 'auto',
    endpoint_seed: int = 42,
) -> MeshCNNSample:
    file_path = Path(obj_path)
    endpoint_order = resolve_endpoint_order(feature_selection.feature_group, endpoint_order)

    obj_mesh = parse_obj(file_path)
    topology = build_topology(obj_mesh, WeldConfig.exact())
    seam_truth = extract_seam_truth(topology)
    if seam_truth.audit.missing_uv_occurrences:
        raise ValueError(
            f'{EXACT_OBJ_LABEL_SOURCE} requires vt indices for every face corner; '
            f'missing occurrences={seam_truth.audit.missing_uv_occurrences}'
        )

    feature_mesh = build_feature_mesh_from_topology(
        topology,
        empty_message='exact OBJ MeshCNN samples require a non-empty triangle mesh',
    )
    edge_features, unique_edges, _ = compute_edge_features_for_selection(
        feature_mesh,
        feature_selection,
        endpoint_order=endpoint_order,
        rng_seed=endpoint_seed,
    )
    feature_names = _feature_names_for_edge_features(feature_selection, edge_features, file_path)
    _assert_exact_edge_order(unique_edges, topology.canonical_edges, file_path)

    faces = np.asarray(feature_mesh.faces, dtype=np.int64)
    unique_edges, edge_to_faces, face_to_edges, edge_neighbors, boundary_mask = build_mesh_adjacency(
        faces,
        unique_edges,
    )
    labels = np.asarray(
        [1.0 if seam_truth.seam_map[(int(edge[0]), int(edge[1]))] else 0.0 for edge in unique_edges],
        dtype=np.float32,
    )

    return MeshCNNSample(
        vertices=torch.from_numpy(np.asarray(feature_mesh.vertices, dtype=np.float32)),
        faces=torch.from_numpy(faces.astype(np.int64)),
        unique_edges=torch.from_numpy(unique_edges.astype(np.int64)),
        edge_features=torch.from_numpy(edge_features.astype(np.float32)),
        edge_labels=torch.from_numpy(labels),
        edge_neighbors=torch.from_numpy(edge_neighbors.astype(np.int64)),
        edge_to_faces=torch.from_numpy(edge_to_faces.astype(np.int64)),
        face_to_edges=torch.from_numpy(face_to_edges.astype(np.int64)),
        boundary_mask=torch.from_numpy(boundary_mask.astype(bool)),
        file_path=str(file_path),
        feature_group=feature_selection.feature_group,
        feature_names=feature_names,
        feature_flags=feature_selection.feature_flags.as_dict(),
        density_config=dict(feature_selection.density_config) if feature_selection.density_config else None,
        endpoint_order=endpoint_order,
        label_source=EXACT_OBJ_LABEL_SOURCE,
        weld_mode=topology.weld_audit.mode,
        seam_edge_count=int(seam_truth.audit.seam_edges),
        boundary_edge_count=int(seam_truth.audit.boundary_edges),
    )


def _mesh_summary(sample: MeshCNNSample) -> dict[str, Any]:
    return {
        'file_path': sample.file_path,
        'vertices': int(sample.vertices.shape[0]),
        'faces': int(sample.faces.shape[0]),
        'unique_edges': int(sample.unique_edges.shape[0]),
        'seam_edges': int(sample.seam_edge_count),
        'boundary_edges': int(sample.boundary_edge_count),
        'feature_dim': int(sample.edge_features.shape[1]),
    }


def build_dataset_manifest(samples: list[MeshCNNSample], dataset_path: Path) -> dict[str, Any]:
    if not samples:
        raise ValueError('cannot build a manifest for an empty dataset')
    first = samples[0]
    summaries = [_mesh_summary(sample) for sample in samples]
    total_edges = sum(item['unique_edges'] for item in summaries)
    total_seams = sum(item['seam_edges'] for item in summaries)
    total_boundary = sum(item['boundary_edges'] for item in summaries)
    total_nonseams = total_edges - total_seams

    manifest = {
        'dataset_path': str(dataset_path),
        'sample_format': SPARSE_MESHCNN_SAMPLE_FORMAT,
        'label_source': EXACT_OBJ_LABEL_SOURCE,
        'feature_group': first.feature_group,
        'feature_names': list(first.feature_names),
        'feature_flags': dict(first.feature_flags),
        'feature_dim': int(first.edge_features.shape[1]),
        'endpoint_order': first.endpoint_order,
        'weld_mode': first.weld_mode,
        'mesh_count': len(samples),
        'total_vertices': sum(item['vertices'] for item in summaries),
        'total_faces': sum(item['faces'] for item in summaries),
        'total_unique_edges': total_edges,
        'total_seam_edges': total_seams,
        'total_boundary_edges': total_boundary,
        'aggregate_seam_ratio': total_seams / max(total_edges, 1),
        'aggregate_pos_weight': total_nonseams / max(total_seams, 1),
        'meshes': summaries,
    }
    if first.density_config is not None:
        manifest['density_config'] = dict(first.density_config)
    return manifest


def write_dataset(samples: list[MeshCNNSample], output_path: Path, overwrite: bool = False) -> Path:
    manifest_path = manifest_path_for_dataset(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f'output exists, pass --overwrite to replace: {output_path}')
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f'manifest exists, pass --overwrite to replace: {manifest_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_saved_meshcnn_feature_metadata(samples)
    torch.save(samples, output_path)
    manifest = build_dataset_manifest(samples, output_path)
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return manifest_path


def resolve_feature_cli_selection(args: argparse.Namespace) -> ResolvedFeatureSet:
    return resolve_feature_selection(
        args.feature_group,
        enable_ao=args.enable_ao,
        enable_dihedral=args.enable_dihedral,
        enable_symmetry=args.enable_symmetry,
        enable_density=args.enable_density,
        enable_thickness_sdf=args.enable_thickness_sdf,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Build isolated MeshCNN edge-segmentation samples from OBJ files.')
    parser.add_argument('mesh_dir', nargs='?', default='./meshes')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-meshes', type=int, default=None)
    parser.add_argument('--feature-group', choices=FEATURE_GROUP_NAMES, default='paper14')
    parser.add_argument('--enable-ao', action='store_true', help='Enable AO endpoint features for custom group')
    parser.add_argument('--enable-dihedral', action='store_true', help='Enable signed dihedral for custom group')
    parser.add_argument('--enable-symmetry', action='store_true', help='Enable symmetry distance for custom group')
    parser.add_argument('--enable-density', action='store_true', help='Enable relative density features for custom group')
    parser.add_argument('--enable-thickness-sdf', action='store_true', help='Enable inward ray thickness for custom group')
    parser.add_argument('--endpoint-order', choices=('auto', *ENDPOINT_ORDERS), default='auto')
    parser.add_argument('--endpoint-seed', type=int, default=42)
    args = parser.parse_args(argv)

    try:
        feature_selection = resolve_feature_cli_selection(args)
    except ValueError as exc:
        parser.error(str(exc))
    endpoint_order = resolve_endpoint_order(feature_selection.feature_group, args.endpoint_order)

    mesh_dir = Path(args.mesh_dir)
    if not mesh_dir.is_dir():
        parser.error(f'directory not found: {mesh_dir}')
    obj_files = sorted(mesh_dir.glob('**/*.obj'))
    if args.max_meshes is not None:
        obj_files = obj_files[:args.max_meshes]
    if not obj_files:
        parser.error(f'no .obj files found in {mesh_dir}')

    print(f'found {len(obj_files)} OBJ file(s)')
    print(f'label source: {EXACT_OBJ_LABEL_SOURCE}')
    print(f'features: {feature_selection.feature_group} ({feature_selection.feature_count})')
    print(f'endpoint order: {endpoint_order}')

    samples: list[MeshCNNSample] = []
    failed = 0
    for idx, obj_file in enumerate(obj_files, start=1):
        try:
            sample = build_meshcnn_sample(
                obj_file,
                feature_selection,
                endpoint_order=endpoint_order,
                endpoint_seed=args.endpoint_seed,
            )
        except Exception as exc:
            failed += 1
            print(f'[{idx}/{len(obj_files)}] skip {obj_file.name}: {exc}')
            continue
        samples.append(sample)
        print(
            f'[{idx}/{len(obj_files)}] {obj_file.name}: '
            f'{sample.num_edges} edges, {sample.seam_edge_count} seams, '
            f'{sample.in_channels} features'
        )

    if not samples:
        raise RuntimeError('no MeshCNN samples were built')

    output_path = Path(args.output)
    manifest_path = write_dataset(samples, output_path, overwrite=args.overwrite)
    manifest = build_dataset_manifest(samples, output_path)
    print(f'saved dataset -> {output_path.resolve()}')
    print(f'saved manifest -> {manifest_path.resolve()}')
    print(
        f'aggregate: {manifest["total_unique_edges"]} edges, '
        f'{manifest["total_seam_edges"]} seams, '
        f'pos_weight {manifest["aggregate_pos_weight"]:.4f}'
    )
    if failed:
        print(f'warning: {failed} file(s) failed')


if __name__ == '__main__':
    main()
