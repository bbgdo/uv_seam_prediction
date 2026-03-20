"""
Inference-time post-processing for seam predictions.

Steps:
  1. threshold_and_clean  — threshold + remove tiny disconnected components
  2. stitch_seam_gaps     — bridge small gaps between seam components (greedy)
  3. postprocess_seams    — combined pipeline

Usage:
    python models/utils/postprocess.py \\
        --dataset dataset.pt \\
        --dual-dataset dataset_dual.pt \\
        --weights runs/dual_graphsage_test/best_model.pth \\
        [--model-type graphsage|gatv2] \\
        [--threshold 0.5] \\
        [--min-component 3] \\
        [--max-gap 3]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def threshold_and_clean(
    probs: np.ndarray,
    unique_edges: np.ndarray,
    threshold: float = 0.5,
    min_component_size: int = 3,
) -> np.ndarray:
    """Threshold probabilities and remove tiny disconnected seam components.

    Returns boolean mask [E] — True = seam edge.
    """
    seam_mask = probs >= threshold
    seam_indices = np.where(seam_mask)[0]

    if len(seam_indices) == 0:
        return seam_mask

    # build vertex -> seam_edge_indices mapping to find shared vertices
    vertex_to_seam: dict[int, list[int]] = {}
    for local_idx, global_idx in enumerate(seam_indices):
        vi, vj = int(unique_edges[global_idx, 0]), int(unique_edges[global_idx, 1])
        vertex_to_seam.setdefault(vi, []).append(local_idx)
        vertex_to_seam.setdefault(vj, []).append(local_idx)

    # adjacency: two seam edges are adjacent if they share a vertex
    n = len(seam_indices)
    rows, cols = [], []
    for incident in vertex_to_seam.values():
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                rows += [incident[i], incident[j]]
                cols += [incident[j], incident[i]]

    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)

    # remove components below size threshold
    comp_sizes = np.bincount(labels, minlength=n_components)
    keep = comp_sizes[labels] >= min_component_size

    cleaned = seam_mask.copy()
    for local_idx, global_idx in enumerate(seam_indices):
        if not keep[local_idx]:
            cleaned[global_idx] = False

    return cleaned


def stitch_seam_gaps(
    probs: np.ndarray,
    seam_mask: np.ndarray,
    unique_edges: np.ndarray,
    edge_to_faces: dict,
    max_gap: int = 3,
) -> np.ndarray:
    """Bridge small gaps between disconnected seam components (greedy).

    For each endpoint vertex of a seam component (vertex with exactly one
    incident seam edge), greedily follow the highest-probability non-seam
    neighbor edges. If we reach another seam component within max_gap steps,
    mark the path as seam.
    """
    mask = seam_mask.copy()

    # build edge lookup structures
    edge_key_to_idx: dict[tuple[int, int], int] = {}
    vertex_to_edges: dict[int, list[int]] = {}
    for idx, (vi, vj) in enumerate(unique_edges):
        vi, vj = int(vi), int(vj)
        key = (min(vi, vj), max(vi, vj))
        edge_key_to_idx[key] = idx
        vertex_to_edges.setdefault(vi, []).append(idx)
        vertex_to_edges.setdefault(vj, []).append(idx)

    def component_label(mask_arr):
        seam_indices = np.where(mask_arr)[0]
        if len(seam_indices) == 0:
            return np.full(len(mask_arr), -1, dtype=np.int32)

        vertex_to_seam: dict[int, list[int]] = {}
        for local_idx, global_idx in enumerate(seam_indices):
            vi, vj = int(unique_edges[global_idx, 0]), int(unique_edges[global_idx, 1])
            vertex_to_seam.setdefault(vi, []).append(local_idx)
            vertex_to_seam.setdefault(vj, []).append(local_idx)

        n = len(seam_indices)
        rows, cols = [], []
        for incident in vertex_to_seam.values():
            for i in range(len(incident)):
                for j in range(i + 1, len(incident)):
                    rows += [incident[i], incident[j]]
                    cols += [incident[j], incident[i]]

        adj = csr_matrix(
            (np.ones(len(rows)), (rows, cols)) if rows else ([], ([], [])),
            shape=(n, n),
        )
        _, labels = connected_components(adj, directed=False)

        # map from global edge idx -> component label (-1 = not seam)
        comp = np.full(len(mask_arr), -1, dtype=np.int32)
        for local_idx, global_idx in enumerate(seam_indices):
            comp[global_idx] = labels[local_idx]
        return comp

    comp = component_label(mask)

    # find endpoint vertices: exactly 1 incident seam edge
    endpoint_verts: set[int] = set()
    for idx in np.where(mask)[0]:
        vi, vj = int(unique_edges[idx, 0]), int(unique_edges[idx, 1])
        for v in (vi, vj):
            seam_count = sum(1 for e in vertex_to_edges.get(v, []) if mask[e])
            if seam_count == 1:
                endpoint_verts.add(v)

    stitched = 0
    for start_v in list(endpoint_verts):
        # find the seam component this endpoint belongs to
        start_edges = [e for e in vertex_to_edges.get(start_v, []) if mask[e]]
        if not start_edges:
            continue
        start_comp = comp[start_edges[0]]

        # greedy walk through non-seam edges, preferring high probability
        path = []
        current_v = start_v
        visited_verts = {start_v}

        for _ in range(max_gap):
            candidates = [
                e for e in vertex_to_edges.get(current_v, [])
                if not mask[e] and e not in path
            ]
            if not candidates:
                break
            # sort by decreasing probability
            candidates.sort(key=lambda e: -probs[e])
            best_edge = candidates[0]
            path.append(best_edge)

            vi, vj = int(unique_edges[best_edge, 0]), int(unique_edges[best_edge, 1])
            next_v = vj if vi == current_v else vi
            if next_v in visited_verts:
                break
            visited_verts.add(next_v)

            # check if we've reached a different seam component
            next_seam_edges = [e for e in vertex_to_edges.get(next_v, []) if mask[e]]
            if next_seam_edges:
                target_comp = comp[next_seam_edges[0]]
                if target_comp != start_comp:
                    # stitch the path
                    for e in path:
                        mask[e] = True
                    stitched += 1
                    break

            current_v = next_v

    return mask


def postprocess_seams(
    probs: np.ndarray,
    unique_edges: np.ndarray,
    edge_to_faces: dict | None = None,
    threshold: float = 0.5,
    min_component_size: int = 3,
    max_gap: int = 3,
) -> np.ndarray:
    """Full post-processing: threshold -> clean -> stitch.

    Returns boolean mask [E] — final seam edges.
    """
    mask = threshold_and_clean(probs, unique_edges, threshold, min_component_size)
    if edge_to_faces is not None:
        mask = stitch_seam_gaps(probs, mask, unique_edges, edge_to_faces, max_gap)
    return mask


def _count_components(mask: np.ndarray, unique_edges: np.ndarray) -> int:
    seam_indices = np.where(mask)[0]
    if len(seam_indices) == 0:
        return 0

    vertex_to_seam: dict[int, list[int]] = {}
    for local_idx, global_idx in enumerate(seam_indices):
        vi, vj = int(unique_edges[global_idx, 0]), int(unique_edges[global_idx, 1])
        vertex_to_seam.setdefault(vi, []).append(local_idx)
        vertex_to_seam.setdefault(vj, []).append(local_idx)

    n = len(seam_indices)
    rows, cols = [], []
    for incident in vertex_to_seam.values():
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                rows += [incident[i], incident[j]]
                cols += [incident[j], incident[i]]

    adj = csr_matrix(
        (np.ones(len(rows)), (rows, cols)) if rows else ([], ([], [])),
        shape=(n, n),
    )
    n_components, labels = connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels, minlength=n_components)
    return n_components, comp_sizes


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    parser = argparse.ArgumentParser(description='Post-process seam predictions.')
    parser.add_argument('--dataset', required=True, help='Path to original dataset.pt (for edge topology)')
    parser.add_argument('--dual-dataset', required=True, help='Path to dual dataset.pt (for model input)')
    parser.add_argument('--weights', required=True, help='Path to best_model.pth')
    parser.add_argument('--model-type', default='graphsage', choices=['graphsage', 'gatv2'],
                        help='Model architecture (default: graphsage)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-component', type=int, default=3)
    parser.add_argument('--max-gap', type=int, default=3)
    parser.add_argument('--mesh-idx', type=int, default=0, help='Which mesh to run on (default: 0)')
    args = parser.parse_args()

    import torch
    from models.utils.dataset import load_dataset

    orig_dataset = load_dataset(args.dataset)
    dual_dataset = load_dataset(args.dual_dataset)

    if args.mesh_idx >= len(orig_dataset):
        print(f"[error] mesh_idx {args.mesh_idx} out of range ({len(orig_dataset)} meshes)")
        sys.exit(1)

    orig_data = orig_dataset[args.mesh_idx]
    dual_data = dual_dataset[args.mesh_idx]

    device = torch.device('cpu')

    if args.model_type == 'graphsage':
        from models.dual_graphsage.model import DualGraphSAGE
        model = DualGraphSAGE().to(device)
    else:
        from models.gatv2.model import DualGATv2
        model = DualGATv2().to(device)

    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(dual_data.x.to(device), dual_data.edge_index.to(device))
    probs = torch.sigmoid(logits).numpy()

    # rebuild edge topology from original data for stitching
    num_unique = dual_data.num_nodes
    src = orig_data.edge_index[0, :num_unique].numpy()
    dst = orig_data.edge_index[1, :num_unique].numpy()
    unique_edges = np.stack([src, dst], axis=1)

    edge_to_faces: dict[tuple, list] = {}
    if hasattr(orig_data, 'faces'):
        faces = orig_data.faces.numpy()
        for f_idx, face in enumerate(faces):
            for k in range(3):
                vi, vj = int(face[k]), int(face[(k + 1) % 3])
                key = (min(vi, vj), max(vi, vj))
                edge_to_faces.setdefault(key, []).append(f_idx)

    # before
    mask_raw = probs >= args.threshold
    n_before = mask_raw.sum()
    n_comp_before, sizes_before = _count_components(mask_raw, unique_edges)
    small_before = (sizes_before < args.min_component).sum() if n_comp_before else 0

    # after
    mask_clean = threshold_and_clean(probs, unique_edges, args.threshold, args.min_component)
    mask_final = stitch_seam_gaps(probs, mask_clean, unique_edges, edge_to_faces, args.max_gap)
    n_after = mask_final.sum()
    n_comp_after, _ = _count_components(mask_final, unique_edges)
    stitched = n_after - mask_clean.sum()

    print(f"mesh: {getattr(orig_data, 'file_path', f'index {args.mesh_idx}')}")
    print(f"before: {n_before} seam edges, {n_comp_before} components ({small_before} with <{args.min_component} edges)")
    print(f"after:  {n_after} seam edges, {n_comp_after} components, {max(stitched, 0)} edges stitched")
