import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

warnings.filterwarnings('ignore', category=UserWarning)
import trimesh  # noqa: E402

# support running both as `python preprocessing/build_gnn_dataset.py` and as a module
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from preprocessing.compute_features import ENDPOINT_ORDERS, FEATURE_PRESETS, compute_edge_features_for_selection
    from preprocessing.feature_registry import FEATURE_GROUP_NAMES, ResolvedFeatureSet, resolve_feature_selection
    from preprocessing.obj_parser import parse_obj
    from preprocessing.seam_labels import extract_seam_truth
    from preprocessing.topology import WeldConfig, build_topology, canonical_edge_key
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from compute_features import ENDPOINT_ORDERS, FEATURE_PRESETS, compute_edge_features_for_selection
    from feature_registry import FEATURE_GROUP_NAMES, ResolvedFeatureSet, resolve_feature_selection
    from obj_parser import parse_obj
    from seam_labels import extract_seam_truth
    from topology import WeldConfig, build_topology, canonical_edge_key

EXACT_DATASET_OUTPUT = 'dataset_v2_exact_labels.pt'


def resolve_endpoint_order(feature_group: str, endpoint_order: str) -> str:
    if endpoint_order != 'auto':
        return endpoint_order
    return 'random' if feature_group == 'paper14' else 'fixed'


def resolve_feature_cli_selection(
    feature_preset: str = 'paper14',
    feature_group: str | None = None,
    enable_ao: bool = False,
    enable_dihedral: bool = False,
    enable_symmetry: bool = False,
    enable_density: bool = False,
    enable_thickness_sdf: bool = False,
) -> ResolvedFeatureSet:
    return resolve_feature_selection(
        feature_group or feature_preset,
        enable_ao=enable_ao,
        enable_dihedral=enable_dihedral,
        enable_symmetry=enable_symmetry,
        enable_density=enable_density,
        enable_thickness_sdf=enable_thickness_sdf,
    )


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
    feature_selection: ResolvedFeatureSet,
    endpoint_order: str,
    label_source: str,
) -> Data:
    feature_names = _feature_names_for_edge_features(feature_selection, edge_features, file_path)
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
    data.feature_preset = feature_selection.feature_preset
    data.feature_group = feature_selection.feature_group
    data.feature_names = feature_names
    data.feature_flags = feature_selection.feature_flags.as_dict()
    if feature_selection.density_config is not None:
        data.density_config = dict(feature_selection.density_config)
    data.endpoint_order = endpoint_order
    data.unique_edges = torch.from_numpy(unique_edges.astype(np.int64))
    data.graph_format = 'primal_mesh_graph'
    return data


def resolve_output_path(output: str | None) -> Path:
    return Path(output) if output is not None else Path(EXACT_DATASET_OUTPUT)


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
        'feature_dim': _saved_graph_feature_dim(data),
    }


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


def _saved_graph_feature_dim(data: Data) -> int:
    x = getattr(data, 'x', None)
    feature_names = list(getattr(data, 'feature_names', []))
    if x is not None and getattr(x, 'ndim', 0) == 2 and len(feature_names) == int(x.shape[1]):
        return int(x.shape[1])

    edge_attr = getattr(data, 'edge_attr', None)
    if edge_attr is not None and getattr(edge_attr, 'ndim', 0) == 2:
        return int(edge_attr.shape[1])

    if x is not None and getattr(x, 'ndim', 0) == 2:
        return int(x.shape[1])

    raise ValueError(f'{getattr(data, "file_path", "<unknown>")}: missing rank-2 feature tensor')


def validate_saved_gnn_feature_metadata(dataset: list[Data]) -> None:
    """Ensure saved PyG graphs expose edge features as data.x with matching names."""
    for graph_idx, data in enumerate(dataset):
        feature_names = list(getattr(data, 'feature_names', []))
        x = getattr(data, 'x', None)
        if x is None or getattr(x, 'ndim', 0) != 2:
            raise ValueError(f'dataset graph {graph_idx} is missing rank-2 x feature tensor')

        feature_dim = int(x.shape[1])
        if len(feature_names) != feature_dim:
            raise ValueError(
                f'dataset graph {graph_idx} feature_names length {len(feature_names)} '
                f'does not match x feature dim {feature_dim}; '
                f'file_path={getattr(data, "file_path", "<unknown>")}, '
                f'feature_names={feature_names}'
            )


def _extract_unique_edges(data: Data) -> np.ndarray:
    unique_edges = getattr(data, 'unique_edges', None)
    if unique_edges is not None:
        if torch.is_tensor(unique_edges):
            unique_edges = unique_edges.detach().cpu().numpy()
        return np.asarray(unique_edges, dtype=np.int64)

    num_directed = int(data.edge_index.shape[1])
    num_unique = num_directed // 2
    return data.edge_index[:, :num_unique].T.detach().cpu().numpy().astype(np.int64, copy=False)


def build_dual_edge_index_from_unique_edges(unique_edges: np.ndarray) -> torch.LongTensor:
    """Build line-graph adjacency for canonical undirected mesh edges."""
    unique_edges = np.asarray(unique_edges, dtype=np.int64)
    if unique_edges.ndim != 2 or unique_edges.shape[1] != 2:
        raise ValueError(f'unique_edges must have shape [E, 2], got {unique_edges.shape}')

    vertex_to_edges: dict[int, list[int]] = {}
    for idx, (vi, vj) in enumerate(unique_edges):
        vertex_to_edges.setdefault(int(vi), []).append(idx)
        vertex_to_edges.setdefault(int(vj), []).append(idx)

    dual_edges_set: set[tuple[int, int]] = set()
    for incident in vertex_to_edges.values():
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                a, b = incident[i], incident[j]
                dual_edges_set.add((a, b))
                dual_edges_set.add((b, a))

    if not dual_edges_set:
        return torch.empty((2, 0), dtype=torch.long)
    dual_edges = np.array(sorted(dual_edges_set), dtype=np.int64).T
    return torch.from_numpy(dual_edges)


def build_dual_data(original_data: Data) -> Data:
    """Convert an original-graph mesh sample into its dual-graph PyG view."""
    unique_edges = _extract_unique_edges(original_data)
    dual_edges = build_dual_edge_index_from_unique_edges(unique_edges)
    num_unique = int(unique_edges.shape[0])
    dual_x = original_data.edge_attr[:num_unique]
    dual_y = original_data.y[:num_unique]

    dual = Data(
        x=dual_x,
        edge_index=dual_edges,
        y=dual_y,
        num_nodes=num_unique,
    )
    dual.file_path = getattr(original_data, 'file_path', '')
    dual.label_source = getattr(original_data, 'label_source', '')
    dual.feature_preset = getattr(original_data, 'feature_preset', '')
    dual.feature_group = getattr(original_data, 'feature_group', getattr(original_data, 'feature_preset', ''))
    dual.feature_names = list(getattr(original_data, 'feature_names', []))
    dual.feature_flags = dict(getattr(original_data, 'feature_flags', {}))
    if hasattr(original_data, 'density_config'):
        dual.density_config = dict(getattr(original_data, 'density_config'))
    dual.endpoint_order = getattr(original_data, 'endpoint_order', '')
    dual.weld_mode = getattr(original_data, 'weld_mode', '')
    dual.seam_edge_count = getattr(original_data, 'seam_edge_count', int(dual_y.sum().item()))
    dual.boundary_edge_count = getattr(original_data, 'boundary_edge_count', 0)
    dual.unique_edges = torch.from_numpy(unique_edges.astype(np.int64))
    dual.graph_format = 'dual_edge_graph'
    return dual


def build_dataset_manifest(dataset: list[Data], dataset_path: Path) -> dict:
    if not dataset:
        raise ValueError('cannot build a manifest for an empty dataset')

    label_source = getattr(dataset[0], 'label_source', '')
    feature_preset = getattr(dataset[0], 'feature_preset', '')
    feature_group = getattr(dataset[0], 'feature_group', feature_preset)
    feature_names = list(getattr(dataset[0], 'feature_names', []))
    feature_flags = dict(getattr(dataset[0], 'feature_flags', {}))
    density_config = getattr(dataset[0], 'density_config', None)
    endpoint_order = getattr(dataset[0], 'endpoint_order', '')
    weld_mode = getattr(dataset[0], 'weld_mode', '')
    graph_format = getattr(dataset[0], 'graph_format', '')

    summaries = [_mesh_summary(data) for data in dataset]
    total_nodes = sum(item['nodes'] for item in summaries)
    total_unique_edges = sum(item['unique_edges'] for item in summaries)
    total_directed_edges = sum(int(data.edge_index.shape[1]) for data in dataset)
    total_seam_edges = sum(item['seam_edges'] for item in summaries)
    total_boundary_edges = sum(item['boundary_edges'] for item in summaries)
    total_nonseam_edges = total_unique_edges - total_seam_edges

    manifest = {
        'dataset_path': str(dataset_path),
        'label_source': label_source,
        'feature_preset': feature_preset,
        'feature_group': feature_group,
        'feature_flags': feature_flags,
        'feature_names': feature_names,
        'endpoint_order': endpoint_order,
        'weld_mode': weld_mode,
        'graph_format': graph_format,
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
    if density_config is not None:
        manifest['density_config'] = dict(density_config)
    return manifest


def write_dataset_manifest(dataset: list[Data], dataset_path: Path) -> Path:
    manifest_path = manifest_path_for_dataset(dataset_path)
    manifest = build_dataset_manifest(dataset, dataset_path)
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return manifest_path


def build_feature_mesh_from_topology(topology) -> trimesh.Trimesh:
    vertices = np.asarray(topology.canonical_vertices, dtype=np.float64)
    faces = np.asarray([face.vertex_ids for face in topology.canonical_faces], dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError('exact_obj requires a non-empty OBJ mesh')
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


_build_feature_mesh_from_topology = build_feature_mesh_from_topology


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
    feature_selection: ResolvedFeatureSet,
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
    feature_mesh = build_feature_mesh_from_topology(topology)

    edge_features, unique_edges, _ = compute_edge_features_for_selection(
        feature_mesh,
        feature_selection,
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
        feature_selection=feature_selection,
        endpoint_order=endpoint_order,
        label_source='exact_obj',
    )
    data.seam_edge_count = int(seam_truth.audit.seam_edges)
    data.boundary_edge_count = int(seam_truth.audit.boundary_edges)
    data.weld_mode = topology.weld_audit.mode
    return data


def process_mesh(
    file_path: str | Path,
    feature_preset: str = 'paper14',
    feature_group: str | None = None,
    enable_ao: bool = False,
    enable_dihedral: bool = False,
    enable_symmetry: bool = False,
    enable_density: bool = False,
    enable_thickness_sdf: bool = False,
    endpoint_order: str = 'auto',
    endpoint_seed: int = 42,
) -> Data | None:
    """Load an .obj file and return a PyG Data object with exact OBJ seam labels."""
    file_path = Path(file_path)
    feature_selection = resolve_feature_cli_selection(
        feature_preset=feature_preset,
        feature_group=feature_group,
        enable_ao=enable_ao,
        enable_dihedral=enable_dihedral,
        enable_symmetry=enable_symmetry,
        enable_density=enable_density,
        enable_thickness_sdf=enable_thickness_sdf,
    )
    endpoint_order = resolve_endpoint_order(feature_selection.feature_group, endpoint_order)
    return _process_mesh_exact_obj(file_path, feature_selection, endpoint_order, endpoint_seed)


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
    parser.add_argument(
        '--feature-preset',
        choices=FEATURE_PRESETS,
        default='paper14',
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--feature-group', choices=FEATURE_GROUP_NAMES, default=None)
    parser.add_argument('--enable-ao', action='store_true', help='Enable AO endpoint features for custom group')
    parser.add_argument('--enable-dihedral', action='store_true', help='Enable signed dihedral for custom group')
    parser.add_argument('--enable-symmetry', action='store_true', help='Enable symmetry distance for custom group')
    parser.add_argument('--enable-density', action='store_true', help='Enable topology-local relative density for custom group')
    parser.add_argument('--enable-thickness-sdf', action='store_true', help='Enable inward ray thickness for custom group')
    parser.add_argument('--endpoint-order', choices=('auto', *ENDPOINT_ORDERS), default='auto')
    parser.add_argument('--endpoint-seed', type=int, default=42)
    args = parser.parse_args(argv)
    try:
        feature_selection = resolve_feature_cli_selection(
            feature_preset=args.feature_preset,
            feature_group=args.feature_group,
            enable_ao=args.enable_ao,
            enable_dihedral=args.enable_dihedral,
            enable_symmetry=args.enable_symmetry,
            enable_density=args.enable_density,
            enable_thickness_sdf=args.enable_thickness_sdf,
        )
    except ValueError as exc:
        parser.error(str(exc))
    endpoint_order = resolve_endpoint_order(feature_selection.feature_group, args.endpoint_order)

    mesh_dir = Path(args.mesh_dir)
    if not mesh_dir.is_dir():
        print(f"[error] directory not found: {mesh_dir}")
        sys.exit(1)

    obj_files = sorted(mesh_dir.glob('**/*.obj'))
    if not obj_files:
        print(f"[error] no .obj files found in {mesh_dir}")
        sys.exit(1)

    print(f"\nfound {len(obj_files)} .obj file(s) in '{mesh_dir}'.")
    print(
        f"features: {feature_selection.feature_group} "
        f"({feature_selection.feature_count}) [{', '.join(feature_selection.feature_names)}]"
    )
    print(f"processing first {min(args.max_meshes, len(obj_files))} ...\n")

    dataset: list[Data] = []
    outliers: list[str] = []
    failed = 0

    for obj_file in obj_files[:args.max_meshes]:
        print(f"processing: {obj_file.name} ...", end=" ", flush=True)
        data = process_mesh(
            obj_file,
            feature_preset=args.feature_preset,
            feature_group=args.feature_group,
            enable_ao=args.enable_ao,
            enable_dihedral=args.enable_dihedral,
            enable_symmetry=args.enable_symmetry,
            enable_density=args.enable_density,
            enable_thickness_sdf=args.enable_thickness_sdf,
            endpoint_order=endpoint_order,
            endpoint_seed=args.endpoint_seed,
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
        print(f"\n  train with: python tools/run_baseline.py --dataset <output.pt> --pos-weight {agg_pos_weight:.4f}")
        print(f"{'#'*60}\n")

    if args.save and dataset:
        out_path = resolve_output_path(args.output)
        if out_path.exists() and not args.overwrite:
            print(f"[error] output exists, pass --overwrite to replace: {out_path}")
            sys.exit(1)
        manifest_path = manifest_path_for_dataset(out_path)
        if manifest_path.exists() and not args.overwrite:
            print(f"[error] manifest exists, pass --overwrite to replace: {manifest_path}")
            sys.exit(1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_to_save = [build_dual_data(data) for data in dataset]
        validate_saved_gnn_feature_metadata(dataset_to_save)
        torch.save(dataset_to_save, out_path)
        print(f"dataset saved -> {out_path.resolve()}  ({len(dataset_to_save)} dual graphs)")
        manifest_path = write_dataset_manifest(dataset_to_save, out_path)
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
