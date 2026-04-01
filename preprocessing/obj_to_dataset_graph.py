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
from compute_features import compute_edge_features


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
            key = (min(vi, vj), max(vi, vj))
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
            # non-manifold — treat as seam
            seam_map[edge] = True

    return seam_map


def process_mesh(file_path: str | Path) -> Data | None:
    """Load an .obj file and return a PyG Data object, or None on failure.

    Detects UV seams on the UV-split topology (as loaded by trimesh),
    then merges duplicate vertices to get the geometric topology that
    matches Blender's representation at inference time. Features are
    computed on the merged mesh.
    """
    from scipy.spatial import cKDTree

    file_path = Path(file_path)

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
        key = (min(geo_vi, geo_vj), max(geo_vi, geo_vj))
        # any split edge being a seam -> geometric edge is a seam
        if is_seam:
            seam_map[key] = True
        elif key not in seam_map:
            seam_map[key] = False

    # 5. Compute features on the merged (geometric) mesh
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vert_nrms = np.asarray(mesh.vertex_normals, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    edge_features, unique_edges, edge_to_faces = compute_edge_features(mesh)

    x = torch.from_numpy(np.concatenate([vertices, vert_nrms], axis=1))

    vi_idx = unique_edges[:, 0]
    vj_idx = unique_edges[:, 1]

    labels = np.array(
        [1.0 if seam_map.get((int(e[0]), int(e[1])), False) else 0.0 for e in unique_edges],
        dtype=np.float32,
    )

    src = np.concatenate([vi_idx, vj_idx])
    dst = np.concatenate([vj_idx, vi_idx])
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0).astype(np.int64))
    edge_attr = torch.from_numpy(np.tile(edge_features, (2, 1)))
    y = torch.from_numpy(np.tile(labels, 2))

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(vertices),
    )
    data.faces = torch.from_numpy(faces)
    data.file_path = str(file_path)
    return data


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
    print(f"  edge features : {data.edge_attr.shape[1]}  (11-dim feature vector)")
    print(f"  --- class balance ---")
    print(f"  seam  (1): {num_seams:>8d}  ({seam_pct:.2f}%)")
    print(f"  other (0): {num_nonseams:>8d}  ({100 - seam_pct:.2f}%)")
    print(f"  pos_weight: {pos_weight:.4f}")

    sample_idx = min(5, data.edge_attr.shape[0] - 1)
    sample = data.edge_attr[sample_idx].numpy()
    print(f"  sample edge_attr[{sample_idx}]: [{', '.join(f'{v:.4f}' for v in sample)}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build PyG UV-seam dataset from .obj files.')
    parser.add_argument('mesh_dir', nargs='?', default='./meshes', help='Directory with .obj files (default: ./meshes)')
    parser.add_argument('--max-meshes', type=int, default=5, help='Max meshes to process (default: 5)')
    parser.add_argument('--save', action='store_true', help='Save dataset as dataset.pt')
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir)
    if not mesh_dir.is_dir():
        print(f"[error] directory not found: {mesh_dir}")
        sys.exit(1)

    obj_files = sorted(mesh_dir.glob('**/*.obj'))
    if not obj_files:
        print(f"[error] no .obj files found in {mesh_dir}")
        sys.exit(1)

    print(f"\nfound {len(obj_files)} .obj file(s) in '{mesh_dir}'.")
    print(f"processing first {min(args.max_meshes, len(obj_files))} …\n")

    dataset: list[Data] = []
    outliers: list[str] = []
    failed = 0

    for obj_file in obj_files[:args.max_meshes]:
        print(f"processing: {obj_file.name} …", end=" ", flush=True)
        data = process_mesh(obj_file)
        if data is None:
            failed += 1
            continue
        print("ok")
        if data.y.sum().item() == 0:
            outliers.append(obj_file.name)
            print(f"  [outlier] {obj_file.name}: 0 seam edges — skipped.")
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
        out_path = Path('dataset.pt')
        torch.save(dataset, out_path)
        print(f"dataset saved -> {out_path.resolve()}  ({len(dataset)} graphs)")

    if outliers:
        print(f"\n{'!'*60}")
        print(f"  outliers — {len(outliers)} file(s) with 0 seam edges (excluded):")
        for name in outliers:
            print(f"    • {name}")
        print(f"{'!'*60}")

    if failed:
        print(f"\n[warning] {failed} file(s) failed to load.")

    print("\ndone.")