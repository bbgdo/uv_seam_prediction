import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

warnings.filterwarnings('ignore', category=UserWarning)
import trimesh  # noqa: E402

# support running both as `python preprocessing/obj_to_dataset_graph.py` and as a module
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from preprocessing.compute_features import ENDPOINT_ORDERS, FEATURE_PRESETS, compute_edge_features
    from preprocessing.obj_parser import parse_obj
    from preprocessing.seam_labels import extract_seam_truth
    from preprocessing.topology import WeldConfig, build_topology, canonical_edge_key
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from compute_features import ENDPOINT_ORDERS, FEATURE_PRESETS, compute_edge_features
    from obj_parser import parse_obj
    from seam_labels import extract_seam_truth
    from topology import WeldConfig, build_topology, canonical_edge_key

LABEL_SOURCES = ('legacy_uv_remap', 'exact_obj')
LEGACY_DATASET_OUTPUT = 'dataset.pt'
EXACT_DATASET_OUTPUT = 'dataset_v2_exact_labels.pt'


def resolve_endpoint_order(feature_preset: str, endpoint_order: str) -> str:
    if endpoint_order != 'auto':
        return endpoint_order
    return 'random' if feature_preset == 'paper14' else 'fixed'


def _detect_seam_edges(mesh: trimesh.Trimesh) -> dict:
    faces = mesh.faces
    has_uv = (
        hasattr(mesh, 'visual')
        and hasattr(mesh.visual, 'uv')
        and mesh.visual.uv is not None
        and len(mesh.visual.uv) > 0
    )

    edge_to_faces: dict[tuple, list] = {}
    for f_idx, face in enumerate(faces):
        for k in range(3):
            vi = face[k]
            vj = face[(k + 1) % 3]
            key = canonical_edge_key(int(vi), int(vj))
            edge_to_faces.setdefault(key, []).append(f_idx)

    seam_map: dict[tuple, bool] = {}

    if not has_uv:
        for edge, face_list in edge_to_faces.items():
            seam_map[edge] = (len(face_list) == 1)
        return seam_map

    uv = mesh.visual.uv
    # trimesh may give UV per face-corner instead of per merged vertex when the mesh has UV splits
    uv_is_per_face_corner = (len(uv) == len(faces) * 3)

    def get_uv_for_vertex_in_face(face_idx: int, geom_vertex: int) -> np.ndarray:
        if uv_is_per_face_corner:
            face = faces[face_idx]
            local_pos = np.where(face == geom_vertex)[0]
            if len(local_pos) == 0:
                return np.array([0.0, 0.0])
            fc_idx = face_idx * 3 + local_pos[0]
            return uv[fc_idx]
        else:
            if geom_vertex < len(uv):
                return uv[geom_vertex]
            return np.array([0.0, 0.0])

    UV_EPS = 1e-5

    for edge, face_list in edge_to_faces.items():
        vi, vj = edge
        if len(face_list) == 1:
            seam_map[edge] = True
        elif len(face_list) == 2:
            f0, f1 = face_list
            uv_vi_f0 = get_uv_for_vertex_in_face(f0, vi)
            uv_vi_f1 = get_uv_for_vertex_in_face(f1, vi)
            uv_vj_f0 = get_uv_for_vertex_in_face(f0, vj)
            uv_vj_f1 = get_uv_for_vertex_in_face(f1, vj)
            split_i = np.linalg.norm(uv_vi_f0 - uv_vi_f1) > UV_EPS
            split_j = np.linalg.norm(uv_vj_f0 - uv_vj_f1) > UV_EPS
            seam_map[edge] = bool(split_i or split_j)
        else:
            # Keep legacy behavior for the transition period.
            seam_map[edge] = True

    return seam_map


def _build_graph_data(
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_features: np.ndarray,
    unique_edges: np.ndarray,
    labels: np.ndarray,
    file_path: Path,
    feature_preset: str,
    endpoint_order: str,
    label_source: str,
) -> Data:
    vert_nrms = np.asarray(mesh.vertex_normals, dtype=np.float32)
    x = torch.from_numpy(np.concatenate([vertices, vert_nrms], axis=1))

    vi_idx = unique_edges[:, 0]
    vj_idx = unique_edges[:, 1]

    src = np.concatenate([vi_idx, vj_idx])
    dst = np.concatenate([vj_idx, vi_idx])
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0).astype(np.int64))
    edge_attr = torch.from_numpy(np.tile(edge_features, (2, 1)))
    y = torch.from_numpy(np.tile(labels.astype(np.float32), 2))

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(vertices),
    )
    data.faces = torch.from_numpy(faces)
    data.file_path = str(file_path)
    data.label_source = label_source
    data.feature_preset = feature_preset
    data.endpoint_order = endpoint_order
    data.unique_edges = torch.from_numpy(unique_edges.astype(np.int64))
    return data


def resolve_output_path(label_source: str, output: str | None) -> Path:
    if output is not None:
        return Path(output)
    if label_source == 'exact_obj':
        return Path(EXACT_DATASET_OUTPUT)
    return Path(LEGACY_DATASET_OUTPUT)


def manifest_path_for_dataset(dataset_path: Path) -> Path:
    return dataset_path.with_name(f'{dataset_path.stem}_manifest.json')


def _unique_edge_count(data: Data) -> int:
    unique_edges = getattr(data, 'unique_edges', None)
    if unique_edges is not None:
        return int(unique_edges.shape[0])
    return int(data.edge_index.shape[1] // 2)


def _unique_labels(data: Data) -> torch.Tensor:
    return data.y[:_unique_edge_count(data)]


def _mesh_summary(data: Data) -> dict:
    unique_edges = _unique_edge_count(data)
    seam_edges = int(getattr(data, 'seam_edge_count', int(_unique_labels(data).sum().item())))
    boundary_edges = int(getattr(data, 'boundary_edge_count', 0))
    return {
        'file_path': getattr(data, 'file_path', ''),
        'nodes': int(data.num_nodes),
        'unique_edges': unique_edges,
        'seam_edges': seam_edges,
        'boundary_edges': boundary_edges,
        'feature_dim': int(data.edge_attr.shape[1]),
    }


def build_dataset_manifest(dataset: list[Data], dataset_path: Path) -> dict:
    if not dataset:
        raise ValueError('cannot build a manifest for an empty dataset')

    label_source = getattr(dataset[0], 'label_source', '')
    feature_preset = getattr(dataset[0], 'feature_preset', '')
    endpoint_order = getattr(dataset[0], 'endpoint_order', '')
    weld_mode = getattr(dataset[0], 'weld_mode', '')

    summaries = [_mesh_summary(data) for data in dataset]
    total_nodes = sum(item['nodes'] for item in summaries)
    total_unique_edges = sum(item['unique_edges'] for item in summaries)
    total_directed_edges = sum(int(data.edge_index.shape[1]) for data in dataset)
    total_seam_edges = sum(item['seam_edges'] for item in summaries)
    total_boundary_edges = sum(item['boundary_edges'] for item in summaries)
    total_nonseam_edges = total_unique_edges - total_seam_edges

    return {
        'dataset_path': str(dataset_path),
        'label_source': label_source,
        'feature_preset': feature_preset,
        'endpoint_order': endpoint_order,
        'weld_mode': weld_mode,
        'mesh_count': len(dataset),
        'total_nodes': total_nodes,
        'total_unique_edges': total_unique_edges,
        'total_directed_edges': total_directed_edges,
        'total_seam_edges': total_seam_edges,
        'total_boundary_edges': total_boundary_edges,
        'aggregate_seam_ratio': total_seam_edges / max(total_unique_edges, 1),
        'aggregate_pos_weight': total_nonseam_edges / max(total_seam_edges, 1),
        'meshes': summaries,
    }


def write_dataset_manifest(dataset: list[Data], dataset_path: Path) -> Path:
    manifest_path = manifest_path_for_dataset(dataset_path)
    manifest = build_dataset_manifest(dataset, dataset_path)
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return manifest_path


def _process_mesh_legacy_uv_remap(
    file_path: Path,
    feature_preset: str,
    endpoint_order: str,
    endpoint_seed: int,
) -> Data | None:
    """Legacy seam labels from trimesh UV splits, vertex merge, and KDTree remap."""
    from scipy.spatial import cKDTree

    try:
        mesh = trimesh.load(str(file_path), process=False, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"  [skip] {file_path.name}: not a single Trimesh object.")
            return None
        if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            print(f"  [skip] {file_path.name}: empty mesh.")
            return None
    except Exception as exc:
        print(f"  [error] {file_path.name}: {exc}")
        return None

    # 1. Detect UV seams on UV-split topology (before merging)
    seam_map_split = _detect_seam_edges(mesh)

    # 2. Merge duplicate vertices to get geometric topology
    split_verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    n_split = len(split_verts)
    mesh.merge_vertices()
    n_merged = len(mesh.vertices)

    if n_split != n_merged:
        print(f"  [merge] {n_split} -> {n_merged} verts ({n_split - n_merged} UV splits removed)")

    # 3. Map split vertex indices to merged vertex indices
    tree = cKDTree(np.asarray(mesh.vertices, dtype=np.float64))
    _, old_to_new = tree.query(split_verts)

    # 4. Remap seam labels to merged edge keys
    seam_map: dict[tuple, bool] = {}
    for (vi, vj), is_seam in seam_map_split.items():
        geo_vi, geo_vj = int(old_to_new[vi]), int(old_to_new[vj])
        if geo_vi == geo_vj:
            continue
        key = canonical_edge_key(geo_vi, geo_vj)
        # any split edge being a seam -> geometric edge is a seam
        if is_seam:
            seam_map[key] = True
        elif key not in seam_map:
            seam_map[key] = False

    # 5. Compute features on the merged (geometric) mesh
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    edge_features, unique_edges, _ = compute_edge_features(
        mesh,
        feature_preset=feature_preset,
        endpoint_order=endpoint_order,
        rng_seed=endpoint_seed,
    )

    labels = np.array(
        [1.0 if seam_map.get((int(e[0]), int(e[1])), False) else 0.0 for e in unique_edges],
        dtype=np.float32,
    )

    return _build_graph_data(
        mesh=mesh,
        vertices=vertices,
        faces=faces,
        edge_features=edge_features,
        unique_edges=unique_edges,
        labels=labels,
        file_path=file_path,
        feature_preset=feature_preset,
        endpoint_order=endpoint_order,
        label_source='legacy_uv_remap',
    )


def _build_feature_mesh_from_topology(topology) -> trimesh.Trimesh:
    vertices = np.asarray(topology.canonical_vertices, dtype=np.float64)
    faces = np.asarray([face.vertex_ids for face in topology.canonical_faces], dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError('exact_obj requires a non-empty OBJ mesh')
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _assert_exact_edge_order(unique_edges: np.ndarray, canonical_edges: tuple, file_path: Path) -> None:
    expected_edges = np.asarray(canonical_edges, dtype=np.int64).reshape((-1, 2))
    if np.array_equal(unique_edges, expected_edges):
        return

    detail = ''
    if unique_edges.shape == expected_edges.shape:
        mismatch_rows = np.flatnonzero(np.any(unique_edges != expected_edges, axis=1))
        if len(mismatch_rows) > 0:
            idx = int(mismatch_rows[0])
            detail = f'; first mismatch at {idx}: features={tuple(unique_edges[idx])}, topology={tuple(expected_edges[idx])}'
    else:
        detail = f'; features shape={unique_edges.shape}, topology shape={expected_edges.shape}'

    raise ValueError(
        f'exact_obj edge order mismatch for {file_path.name}: '
        f'feature_edges={len(unique_edges)}, topology_edges={len(expected_edges)}{detail}'
    )


def _process_mesh_exact_obj(
    file_path: Path,
    feature_preset: str,
    endpoint_order: str,
    endpoint_seed: int,
) -> Data:
    obj_mesh = parse_obj(file_path)
    topology = build_topology(obj_mesh, WeldConfig.exact())
    seam_truth = extract_seam_truth(topology)
    if seam_truth.audit.missing_uv_occurrences:
        raise ValueError(
            f'exact_obj requires vt indices for every face corner; '
            f'missing occurrences={seam_truth.audit.missing_uv_occurrences}'
        )
    feature_mesh = _build_feature_mesh_from_topology(topology)

    edge_features, unique_edges, _ = compute_edge_features(
        feature_mesh,
        feature_preset=feature_preset,
        endpoint_order=endpoint_order,
        rng_seed=endpoint_seed,
    )
    _assert_exact_edge_order(unique_edges, topology.canonical_edges, file_path)

    labels = np.array(
        [1.0 if seam_truth.seam_map[(int(e[0]), int(e[1]))] else 0.0 for e in unique_edges],
        dtype=np.float32,
    )

    data = _build_graph_data(
        mesh=feature_mesh,
        vertices=np.asarray(feature_mesh.vertices, dtype=np.float32),
        faces=np.asarray(feature_mesh.faces, dtype=np.int64),
        edge_features=edge_features,
        unique_edges=unique_edges,
        labels=labels,
        file_path=file_path,
        feature_preset=feature_preset,
        endpoint_order=endpoint_order,
        label_source='exact_obj',
    )
    data.seam_edge_count = int(seam_truth.audit.seam_edges)
    data.boundary_edge_count = int(seam_truth.audit.boundary_edges)
    data.weld_mode = topology.weld_audit.mode
    return data


def process_mesh(
    file_path: str | Path,
    feature_preset: str = 'extended18',
    endpoint_order: str = 'auto',
    endpoint_seed: int = 42,
    label_source: str = 'legacy_uv_remap',
) -> Data | None:
    """Load an .obj file and return a PyG Data object.

    The default label source preserves the legacy trimesh UV-remap behavior.
    exact_obj derives labels from parsed OBJ face-corner topology and uses
    trimesh only for geometric feature computation.
    """
    if label_source not in LABEL_SOURCES:
        raise ValueError(f"label_source must be one of {LABEL_SOURCES}, got: {label_source}")

    file_path = Path(file_path)
    endpoint_order = resolve_endpoint_order(feature_preset, endpoint_order)

    if label_source == 'exact_obj':
        return _process_mesh_exact_obj(file_path, feature_preset, endpoint_order, endpoint_seed)
    return _process_mesh_legacy_uv_remap(file_path, feature_preset, endpoint_order, endpoint_seed)


def print_stats(data: Data, file_name: str) -> None:
    num_edges = data.edge_index.shape[1]
    num_unique_edges = num_edges // 2

    num_seams = data.y.bool().sum().item()
    num_nonseams = num_edges - num_seams
    seam_pct = 100.0 * num_seams / max(num_edges, 1)
    pos_weight = num_nonseams / max(num_seams, 1)

    print(f"\n{'='*60}")
    print(f"  file          : {file_name}")
    print(f"  nodes         : {data.num_nodes}")
    print(f"  unique edges  : {num_unique_edges}")
    print(f"  directed edges: {num_edges}  (both directions)")
    print(f"  edge features : {data.edge_attr.shape[1]}  ({data.edge_attr.shape[1]}-dim feature vector)")
    print(f"  --- class balance ---")
    print(f"  seam  (1): {num_seams:>8d}  ({seam_pct:.2f}%)")
    print(f"  other (0): {num_nonseams:>8d}  ({100 - seam_pct:.2f}%)")
    print(f"  pos_weight: {pos_weight:.4f}")

    sample_idx = min(5, data.edge_attr.shape[0] - 1)
    sample = data.edge_attr[sample_idx].numpy()
    print(f"  sample edge_attr[{sample_idx}]: [{', '.join(f'{v:.4f}' for v in sample)}]")
    print(f"{'='*60}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Build PyG UV-seam dataset from .obj files.')
    parser.add_argument('mesh_dir', nargs='?', default='./meshes', help='Directory with .obj files (default: ./meshes)')
    parser.add_argument('--max-meshes', type=int, default=5, help='Max meshes to process (default: 5)')
    parser.add_argument('--save', action='store_true', help='Save the dataset')
    parser.add_argument(
        '--output',
        default=None,
        help='Output path when --save is set; exact_obj defaults to dataset_v2_exact_labels.pt',
    )
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output file')
    parser.add_argument('--feature-preset', choices=FEATURE_PRESETS, default='extended18')
    parser.add_argument('--endpoint-order', choices=('auto', *ENDPOINT_ORDERS), default='auto')
    parser.add_argument('--endpoint-seed', type=int, default=42)
    parser.add_argument(
        '--label-source',
        choices=LABEL_SOURCES,
        default='legacy_uv_remap',
        help='Seam label source: legacy trimesh UV remap or exact OBJ face-corner topology',
    )
    args = parser.parse_args(argv)
    endpoint_order = resolve_endpoint_order(args.feature_preset, args.endpoint_order)

    mesh_dir = Path(args.mesh_dir)
    if not mesh_dir.is_dir():
        print(f"[error] directory not found: {mesh_dir}")
        sys.exit(1)

    obj_files = sorted(mesh_dir.glob('**/*.obj'))
    if not obj_files:
        print(f"[error] no .obj files found in {mesh_dir}")
        sys.exit(1)

    print(f"\nfound {len(obj_files)} .obj file(s) in '{mesh_dir}'.")
    print(f"label source: {args.label_source}")
    print(f"processing first {min(args.max_meshes, len(obj_files))} ...\n")

    dataset: list[Data] = []
    outliers: list[str] = []
    failed = 0

    for obj_file in obj_files[:args.max_meshes]:
        print(f"processing: {obj_file.name} ...", end=" ", flush=True)
        data = process_mesh(
            obj_file,
            feature_preset=args.feature_preset,
            endpoint_order=endpoint_order,
            endpoint_seed=args.endpoint_seed,
            label_source=args.label_source,
        )
        if data is None:
            failed += 1
            continue
        print("ok")
        if data.y.sum().item() == 0:
            outliers.append(obj_file.name)
            print(f"  [outlier] {obj_file.name}: 0 seam edges - skipped.")
            continue
        dataset.append(data)
        print_stats(data, obj_file.name)

    if dataset:
        total_nodes = sum(d.num_nodes for d in dataset)
        total_edges = sum(d.edge_index.shape[1] for d in dataset)
        total_seams = sum(d.y.sum().item() for d in dataset)
        total_nonseam = total_edges - total_seams
        agg_pos_weight = total_nonseam / max(total_seams, 1)

        print(f"\n{'#'*60}")
        print(f"  aggregate over {len(dataset)} mesh(es)")
        print(f"  total nodes         : {total_nodes}")
        print(f"  total directed edges: {total_edges}")
        print(f"  total seam edges    : {int(total_seams)}  ({100*total_seams/max(total_edges,1):.2f}%)")
        print(f"  aggregate pos_weight: {agg_pos_weight:.4f}")
        print(f"\n  use in training:")
        print(f"      pos_weight = torch.tensor([{agg_pos_weight:.4f}])")
        print(f"      criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)")
        print(f"{'#'*60}\n")

    if args.save and dataset:
        out_path = resolve_output_path(args.label_source, args.output)
        if out_path.exists() and not args.overwrite:
            print(f"[error] output exists, pass --overwrite to replace: {out_path}")
            sys.exit(1)
        manifest_path = manifest_path_for_dataset(out_path)
        if args.label_source == 'exact_obj' and manifest_path.exists() and not args.overwrite:
            print(f"[error] manifest exists, pass --overwrite to replace: {manifest_path}")
            sys.exit(1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, out_path)
        print(f"dataset saved -> {out_path.resolve()}  ({len(dataset)} graphs)")
        if args.label_source == 'exact_obj':
            manifest_path = write_dataset_manifest(dataset, out_path)
            print(f"manifest saved -> {manifest_path.resolve()}")
            print(
                "sanity check: "
                f"python tools/validate_seam_truth.py --mesh-dir {mesh_dir} --max-meshes {len(dataset)}"
            )

    if outliers:
        print(f"\n{'!'*60}")
        print(f"  outliers - {len(outliers)} file(s) with 0 seam edges (excluded):")
        for name in outliers:
            print(f"    - {name}")
        print(f"{'!'*60}")

    if failed:
        print(f"\n[warning] {failed} file(s) failed to load.")

    print("\ndone.")


if __name__ == "__main__":
    main()
