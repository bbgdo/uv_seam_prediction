import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


def build_dual_graph_data(original_data: Data) -> Data:
    """Convert an original-graph Data object into a dual-graph Data object.

    Input Data fields used:
        edge_index: [2, 2E]  — first E columns are unique (vi, vj), second E are reverse
        edge_attr:  [2E, 16] — edge features (first E rows = unique)
        y:          [2E]     — edge labels (first E rows = unique)
        faces:      [F, 3]   — triangle face indices

    Output Data:
        x:          [E, 16]  — dual node features = original edge features
        edge_index: [2, D]   — dual graph connectivity (bidirectional)
        y:          [E]      — dual node labels = original edge labels
    """
    num_directed = original_data.edge_index.shape[1]
    num_unique = num_directed // 2

    src = original_data.edge_index[0, :num_unique].numpy()
    dst = original_data.edge_index[1, :num_unique].numpy()

    # map (min(vi,vj), max(vi,vj)) -> edge_idx
    edge_key_to_idx: dict[tuple, int] = {}
    for idx in range(num_unique):
        vi, vj = int(src[idx]), int(dst[idx])
        key = (min(vi, vj), max(vi, vj))
        edge_key_to_idx[key] = idx

    faces = original_data.faces.numpy()
    dual_edges_set: set[tuple[int, int]] = set()

    for face in faces:
        face_edge_indices = []
        for k in range(3):
            vi, vj = int(face[k]), int(face[(k + 1) % 3])
            key = (min(vi, vj), max(vi, vj))
            if key in edge_key_to_idx:
                face_edge_indices.append(edge_key_to_idx[key])

        # each pair of edges in this face -> dual edge (bidirectional)
        for i in range(len(face_edge_indices)):
            for j in range(i + 1, len(face_edge_indices)):
                a, b = face_edge_indices[i], face_edge_indices[j]
                dual_edges_set.add((a, b))
                dual_edges_set.add((b, a))

    dual_edges = np.array(sorted(dual_edges_set), dtype=np.int64).T  # [2, D]

    dual_x = original_data.edge_attr[:num_unique]
    dual_y = original_data.y[:num_unique]

    dual = Data(
        x=dual_x,
        edge_index=torch.from_numpy(dual_edges),
        y=dual_y,
        num_nodes=num_unique,
    )
    dual.file_path = getattr(original_data, 'file_path', '')
    return dual


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dual graph dataset from original graph dataset.')
    parser.add_argument('--input', required=True, help='Path to original dataset.pt')
    parser.add_argument('--output', required=True, help='Path to save dual dataset (e.g. dataset_dual.pt)')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] not found: {input_path}")
        sys.exit(1)

    print(f"loading {input_path} ...")
    original_dataset = torch.load(input_path, weights_only=False)

    dual_dataset = []
    for i, data in enumerate(original_dataset):
        dual = build_dual_graph_data(data)
        file_name = getattr(data, 'file_path', f'graph_{i}')
        if isinstance(file_name, str):
            file_name = Path(file_name).name

        orig_nodes = data.num_nodes
        orig_edges = data.edge_index.shape[1] // 2
        dual_nodes = dual.num_nodes
        dual_edges = dual.edge_index.shape[1]
        avg_degree = dual_edges / max(dual_nodes, 1)

        print(
            f"  {file_name}: original {orig_nodes} nodes, {orig_edges} edges "
            f"-> dual {dual_nodes} nodes, {dual_edges} edges (avg degree {avg_degree:.1f})"
        )
        dual_dataset.append(dual)

    output_path = Path(args.output)
    torch.save(dual_dataset, output_path)
    print(f"\nsaved {len(dual_dataset)} dual graphs -> {output_path}")
