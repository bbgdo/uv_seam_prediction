import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


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


def build_dual_graph_data(original_data: Data) -> Data:
    """Convert an original-graph Data object into a dual-graph Data object.

    Uses line graph (vertex-sharing) adjacency: two edges are connected in the
    dual when they share a vertex. This matches GraphSeam Section 4.2 and gives
    avg degree ~10 vs ~4 for face-adjacency, providing 2.5x denser receptive field.

    Input Data fields used:
        edge_index: [2, 2E]  — first E columns are unique (vi, vj), second E are reverse
        edge_attr:  [2E, 18] — edge features (first E rows = unique)
        y:          [2E]     — edge labels (first E rows = unique)
        faces:      [F, 3]   — triangle face indices (kept in original data, not used here)

    Output Data:
        x:          [E, 18]  — dual node features = original edge features
        edge_index: [2, D]   — dual graph connectivity (bidirectional, vertex-sharing)
        y:          [E]      — dual node labels = original edge labels
    """
    num_directed = original_data.edge_index.shape[1]
    num_unique = num_directed // 2

    unique_edges = original_data.edge_index[:, :num_unique].T.numpy()
    dual_edges = build_dual_edge_index_from_unique_edges(unique_edges)

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
