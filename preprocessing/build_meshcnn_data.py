import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_edge_neighbors(
    src: np.ndarray,
    dst: np.ndarray,
    faces: np.ndarray,
    edge_key_to_idx: dict,
) -> np.ndarray:
    """Compute [E, 4] neighbor index array for MeshConv.

    For each edge (a, b):
      Face 1 (a, b, c) contributes neighbors at positions 0, 1: (a,c) and (c,b)
      Face 2 (a, b, d) contributes neighbors at positions 2, 3: (b,d) and (d,a)
    Missing neighbors (boundary) are -1.
    """
    num_unique = len(src)

    # build edge → incident face list
    edge_to_faces: dict = {}
    for f_idx, face in enumerate(faces):
        for k in range(3):
            vi, vj = int(face[k]), int(face[(k + 1) % 3])
            key = (min(vi, vj), max(vi, vj))
            edge_to_faces.setdefault(edge_key_to_idx.get(key, -1), []).append(f_idx)
    edge_to_faces.pop(-1, None)

    neighbors = np.full((num_unique, 4), -1, dtype=np.int64)

    for edge_idx in range(num_unique):
        a, b = int(src[edge_idx]), int(dst[edge_idx])
        incident_faces = edge_to_faces.get(edge_idx, [])

        # collect the vertex opposite to (a,b) in each incident face
        opposite_verts: list[int] = []
        for f_idx in incident_faces[:2]:
            face_set = {int(v) for v in faces[f_idx]}
            opp = face_set - {a, b}
            if opp:
                opposite_verts.append(next(iter(opp)))

        if len(opposite_verts) >= 1:
            c = opposite_verts[0]
            neighbors[edge_idx, 0] = edge_key_to_idx.get((min(a, c), max(a, c)), -1)
            neighbors[edge_idx, 1] = edge_key_to_idx.get((min(c, b), max(c, b)), -1)

        if len(opposite_verts) >= 2:
            d = opposite_verts[1]
            neighbors[edge_idx, 2] = edge_key_to_idx.get((min(b, d), max(b, d)), -1)
            neighbors[edge_idx, 3] = edge_key_to_idx.get((min(d, a), max(d, a)), -1)

    return neighbors


def _load_faces(data: Data) -> np.ndarray:
    if hasattr(data, 'faces') and data.faces is not None:
        return data.faces.numpy().astype(np.int64)

    file_path = getattr(data, 'file_path', '')
    if not file_path or not Path(file_path).exists():
        raise ValueError(
            f'Data has no .faces attribute and file_path is missing or not found: {file_path!r}. '
            'Re-run obj_to_dataset_graph.py to rebuild dataset.pt with face data.'
        )

    import trimesh
    mesh = trimesh.load(str(file_path), process=False, force='mesh')
    return np.asarray(mesh.faces, dtype=np.int64)


def build_meshcnn_data(original_data: Data) -> Data:
    """Convert a standard PyG Data object to MeshCNN-compatible format.

    The output uses the same field names as other datasets for compatibility
    with shared utilities: x (features), y (labels), plus edge_neighbors.
    """
    num_directed = original_data.edge_index.shape[1]
    num_unique = num_directed // 2

    # unique edges are stored in the first half of edge_index (forward direction)
    src = original_data.edge_index[0, :num_unique].numpy()
    dst = original_data.edge_index[1, :num_unique].numpy()
    faces = _load_faces(original_data)

    edge_key_to_idx: dict = {}
    for idx in range(num_unique):
        key = (int(min(src[idx], dst[idx])), int(max(src[idx], dst[idx])))
        edge_key_to_idx[key] = idx

    neighbors = build_edge_neighbors(src, dst, faces, edge_key_to_idx)

    edge_feats = original_data.edge_attr[:num_unique].clone()  # [E, F] — F=11 expected
    if edge_feats.shape[1] != 11:
        import warnings
        warnings.warn(
            f'edge_attr has {edge_feats.shape[1]} features, expected 11. '
            'Re-run obj_to_dataset_graph.py to rebuild dataset.pt with 11-dim features.'
        )

    return Data(
        x=edge_feats,                                           # [E, F]
        edge_neighbors=torch.from_numpy(neighbors),             # [E, 4]
        y=original_data.y[:num_unique].clone(),                 # [E]
        num_nodes=num_unique,
        file_path=getattr(original_data, 'file_path', ''),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build MeshCNN-compatible dataset with 4-neighbor structure.'
    )
    parser.add_argument('--input', default='dataset.pt', help='Input dataset.pt path')
    parser.add_argument('--output', default='dataset_meshcnn.pt', help='Output dataset path')
    args = parser.parse_args()

    print(f'loading {args.input}...')
    dataset = torch.load(args.input, weights_only=False)
    print(f'  {len(dataset)} graphs')

    meshcnn_dataset = []
    total_edges = 0
    total_full_neighbors = 0
    total_boundary = 0

    for i, data in enumerate(dataset):
        mcnn_data = build_meshcnn_data(data)
        meshcnn_dataset.append(mcnn_data)

        n_edges = mcnn_data.x.shape[0]
        full = int((mcnn_data.edge_neighbors >= 0).all(dim=1).sum().item())
        boundary = n_edges - full

        total_edges += n_edges
        total_full_neighbors += full
        total_boundary += boundary

        if (i + 1) % 10 == 0 or i == 0:
            name = Path(getattr(data, 'file_path', f'graph_{i}')).name
            print(f'  [{i + 1:3d}/{len(dataset)}] {name}: '
                  f'{n_edges} edges, {boundary} boundary ({100 * boundary / max(n_edges, 1):.1f}%)')

    print(f'\nall graphs:')
    print(f'  total edges: {total_edges}')
    print(f'  interior (4 neighbors): {total_full_neighbors} '
          f'({100 * total_full_neighbors / max(total_edges, 1):.1f}%)')
    print(f'  boundary (< 4 neighbors): {total_boundary} '
          f'({100 * total_boundary / max(total_edges, 1):.1f}%)')

    torch.save(meshcnn_dataset, args.output)
    print(f'\nsaved -> {args.output}')

    # quick schema check
    d = meshcnn_dataset[0]
    print(f'\nfirst graph schema:')
    print(f'  x:              {tuple(d.x.shape)}  (edge features)')
    print(f'  edge_neighbors: {tuple(d.edge_neighbors.shape)}  (4 neighbors per edge)')
    print(f'  y:              {tuple(d.y.shape)}  (seam labels)')
    print(f'  seam ratio:     {d.y.float().mean().item():.4f}')


if __name__ == '__main__':
    main()
