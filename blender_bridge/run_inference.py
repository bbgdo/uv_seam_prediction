"""
Standalone UV seam inference worker — runs in an external Python process.

Usage:
    python run_inference.py <data.npz> <weights.pth> <threshold> <output.txt>
                           [--model-type graphsage|gatv2]
                           [--min-component N]
                           [--max-gap N]

Reads raw mesh geometry from <data.npz>, computes 11-dim edge features,
builds the dual graph, runs the GNN, applies post-processing, and writes
predicted seam edge indices to <output.txt>, one index per line.

This script is fully self-contained — it does not import from the project tree.
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.nn import GATv2Conv, SAGEConv


# ─── Geometry helpers ────────────────────────────────────────────────────────

def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norms < eps, eps, norms)


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    return fn / np.where(norms < 1e-8, 1e-8, norms)


def _build_edge_to_faces(faces: np.ndarray) -> dict:
    edge_to_faces: dict = {}
    for f_idx, face in enumerate(faces):
        for k in range(3):
            vi, vj = int(face[k]), int(face[(k + 1) % 3])
            key = (min(vi, vj), max(vi, vj))
            edge_to_faces.setdefault(key, []).append(f_idx)
    return edge_to_faces


# ─── Feature computation (pure numpy, no trimesh) ────────────────────────────

def _feat_edge_length(vertices: np.ndarray, unique_edges: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(
        vertices[unique_edges[:, 1]] - vertices[unique_edges[:, 0]], axis=1
    )
    return (lengths / (lengths.max() + 1e-8)).astype(np.float32)


def _feat_signed_dihedral(
    vertices: np.ndarray,
    faces: np.ndarray,
    unique_edges: np.ndarray,
    edge_to_faces: dict,
) -> np.ndarray:
    face_normals = _compute_face_normals(vertices, faces)
    angles = np.zeros(len(unique_edges), dtype=np.float32)

    for idx, (vi, vj) in enumerate(unique_edges):
        key = (int(vi), int(vj))
        face_list = edge_to_faces.get(key, [])
        if len(face_list) < 2:
            continue

        n0 = face_normals[face_list[0]].astype(np.float64)
        n1 = face_normals[face_list[1]].astype(np.float64)
        cos_a = np.clip(np.dot(n0, n1), -1.0, 1.0)
        unsigned_angle = np.arccos(cos_a)

        edge_dir = vertices[vj] - vertices[vi]
        en = np.linalg.norm(edge_dir)
        if en > 1e-8:
            edge_dir /= en
            cross = np.cross(n0, n1)
            angles[idx] = float(unsigned_angle * np.sign(np.dot(cross, edge_dir) + 1e-12))
        else:
            angles[idx] = float(unsigned_angle)

    return (angles / np.pi).astype(np.float32)


def _feat_delta_normal(normals: np.ndarray, unique_edges: np.ndarray) -> np.ndarray:
    delta = np.linalg.norm(
        normals[unique_edges[:, 0]] - normals[unique_edges[:, 1]], axis=1
    )
    return (delta / 2.0).astype(np.float32)


def _feat_dot_normal(normals: np.ndarray, unique_edges: np.ndarray) -> np.ndarray:
    n_vi = _safe_normalize(normals[unique_edges[:, 0]])
    n_vj = _safe_normalize(normals[unique_edges[:, 1]])
    return np.einsum('ij,ij->i', n_vi, n_vj).astype(np.float32)


def _vertex_gaussian_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n_verts = len(vertices)
    angle_sum = np.zeros(n_verts, dtype=np.float64)

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        for local_idx, (e_a, e_b) in enumerate([
            (v1 - v0, v2 - v0),
            (v0 - v1, v2 - v1),
            (v0 - v2, v1 - v2),
        ]):
            na, nb = np.linalg.norm(e_a), np.linalg.norm(e_b)
            if na < 1e-12 or nb < 1e-12:
                continue
            cos_a = np.clip(np.dot(e_a, e_b) / (na * nb), -1.0, 1.0)
            angle_sum[face[local_idx]] += np.arccos(cos_a)

    edge_face_count: dict = {}
    for face in faces:
        for k in range(3):
            key = (min(int(face[k]), int(face[(k + 1) % 3])),
                   max(int(face[k]), int(face[(k + 1) % 3])))
            edge_face_count[key] = edge_face_count.get(key, 0) + 1

    boundary_verts: set = set()
    for (vi, vj), count in edge_face_count.items():
        if count == 1:
            boundary_verts.add(vi)
            boundary_verts.add(vj)

    curvatures = np.zeros(n_verts, dtype=np.float64)
    for v in range(n_verts):
        curvatures[v] = (np.pi if v in boundary_verts else 2 * np.pi) - angle_sum[v]

    return curvatures.astype(np.float32)


def _zscore_clip_normalize(values: np.ndarray, clip_range: float = 3.0) -> np.ndarray:
    z = (values - values.mean()) / (values.std() + 1e-8)
    z = np.clip(z, -clip_range, clip_range)
    return (z / clip_range).astype(np.float32)


def _feat_gauss_curvature(
    vertices: np.ndarray, faces: np.ndarray, unique_edges: np.ndarray
) -> tuple:
    k = _vertex_gaussian_curvature(vertices, faces)
    k_n = _zscore_clip_normalize(k)
    k_vi, k_vj = k_n[unique_edges[:, 0]], k_n[unique_edges[:, 1]]
    gauss_mean = ((k_vi + k_vj) / 2.0).astype(np.float32)
    gauss_diff = np.abs(k_vi - k_vj).astype(np.float32)
    return gauss_mean, gauss_diff


def _feat_ao(
    vertices: np.ndarray, normals: np.ndarray, unique_edges: np.ndarray
) -> tuple:
    """Normal-based AO approximation (no raycasting required)."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        ao = np.full(len(vertices), 0.5, dtype=np.float32)
        ao_vi, ao_vj = ao[unique_edges[:, 0]], ao[unique_edges[:, 1]]
        return (ao_vi + ao_vj) / 2, np.abs(ao_vi - ao_vj)

    bbox_diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    radius = bbox_diag * 0.1
    tree = cKDTree(vertices)
    ao = np.zeros(len(vertices), dtype=np.float32)

    for i in range(len(vertices)):
        neighbors = tree.query_ball_point(vertices[i], radius)
        neighbor_idx = [n for n in neighbors if n != i]
        if not neighbor_idx:
            continue
        dirs = vertices[i] - vertices[neighbor_idx]
        dirs_norm = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        alignment = np.einsum('ij,ij->i', normals[neighbor_idx], dirs_norm)
        ao[i] = float(np.clip(np.mean(np.maximum(alignment, 0)), 0, 1))

    ao_vi, ao_vj = ao[unique_edges[:, 0]], ao[unique_edges[:, 1]]
    return ((ao_vi + ao_vj) / 2).astype(np.float32), np.abs(ao_vi - ao_vj).astype(np.float32)


def _feat_symmetry(vertices: np.ndarray, unique_edges: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return np.full(len(unique_edges), 0.5, dtype=np.float32)

    bbox_diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    match_tol = bbox_diag * 0.01
    tree = cKDTree(vertices)
    best_axis = None
    best_ratio = 0.0

    for axis in range(3):
        reflected = vertices.copy()
        reflected[:, axis] *= -1
        dists, _ = tree.query(reflected)
        ratio = float(np.mean(dists < match_tol))
        if ratio > best_ratio:
            best_ratio = ratio
            best_axis = axis

    if best_ratio < 0.8:
        return np.full(len(unique_edges), 0.5, dtype=np.float32)

    midpoints = (vertices[unique_edges[:, 0]] + vertices[unique_edges[:, 1]]) / 2.0
    max_extent = float(np.abs(vertices[:, best_axis]).max()) + 1e-8
    return (np.abs(midpoints[:, best_axis]) / max_extent).astype(np.float32)


def compute_edge_features(
    vertices: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    unique_edges: np.ndarray,
) -> np.ndarray:
    """Compute all 11 edge features from raw mesh arrays.

    Returns:
        features: [E, 11] float32
    """
    edge_to_faces = _build_edge_to_faces(faces)

    f0 = _feat_edge_length(vertices, unique_edges)
    f1 = _feat_signed_dihedral(vertices, faces, unique_edges, edge_to_faces)
    f2 = np.abs(f1)   # sharpness
    f3 = f1.copy()    # concavity
    f4 = _feat_delta_normal(normals, unique_edges)
    f5 = _feat_dot_normal(normals, unique_edges)
    f6, f7 = _feat_gauss_curvature(vertices, faces, unique_edges)
    f8, f9 = _feat_ao(vertices, normals, unique_edges)
    f10 = _feat_symmetry(vertices, unique_edges)

    return np.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], axis=1)


# ─── Dual graph builder ──────────────────────────────────────────────────────

def build_dual_edge_index(faces: np.ndarray, unique_edges: np.ndarray) -> torch.Tensor:
    """Build dual graph edge_index [2, D] from face adjacency.

    Each original edge is a dual node. Two dual nodes are connected when their
    edges share a triangular face.
    """
    edge_key_to_idx: dict = {
        (int(vi), int(vj)): idx for idx, (vi, vj) in enumerate(unique_edges)
    }

    dual_src, dual_dst = [], []
    for face in faces:
        edge_indices = []
        for k in range(3):
            vi, vj = int(face[k]), int(face[(k + 1) % 3])
            key = (min(vi, vj), max(vi, vj))
            if key in edge_key_to_idx:
                edge_indices.append(edge_key_to_idx[key])

        for i in range(len(edge_indices)):
            for j in range(len(edge_indices)):
                if i != j:
                    dual_src.append(edge_indices[i])
                    dual_dst.append(edge_indices[j])

    if not dual_src:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor([dual_src, dual_dst], dtype=torch.long)


# ─── Embedded model definitions ──────────────────────────────────────────────

class DualGraphSAGE(nn.Module):
    def __init__(self, in_dim=11, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if i > 0 and h.shape == x.shape:
                h = h + x
            x = h

        return self.classifier(x).squeeze(-1)


class DualGATv2(nn.Module):
    def __init__(self, in_dim=11, hidden_dim=64, heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        self.convs.append(
            GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if i > 0 and h.shape == x.shape:
                h = h + x
            x = h

        return self.classifier(x).squeeze(-1)


# ─── Post-processing ─────────────────────────────────────────────────────────

def _seam_component_labels(mask: np.ndarray, unique_edges: np.ndarray) -> np.ndarray:
    seam_idx = np.where(mask)[0]
    if len(seam_idx) == 0:
        return np.full(len(mask), -1, dtype=np.int32)

    vertex_to_seam: dict = {}
    for local_idx, global_idx in enumerate(seam_idx):
        vi, vj = int(unique_edges[global_idx, 0]), int(unique_edges[global_idx, 1])
        vertex_to_seam.setdefault(vi, []).append(local_idx)
        vertex_to_seam.setdefault(vj, []).append(local_idx)

    n = len(seam_idx)
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

    comp = np.full(len(mask), -1, dtype=np.int32)
    for local_idx, global_idx in enumerate(seam_idx):
        comp[global_idx] = labels[local_idx]
    return comp


def threshold_and_clean(
    probs: np.ndarray,
    unique_edges: np.ndarray,
    threshold: float = 0.5,
    min_component_size: int = 3,
) -> np.ndarray:
    """Threshold probabilities and remove tiny disconnected seam components."""
    seam_mask = probs >= threshold
    seam_indices = np.where(seam_mask)[0]
    if len(seam_indices) == 0:
        return seam_mask

    vertex_to_seam: dict = {}
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

    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)
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
    max_gap: int = 3,
) -> np.ndarray:
    """Bridge small gaps between disconnected seam components (greedy)."""
    mask = seam_mask.copy()

    vertex_to_edges: dict = {}
    for idx, (vi, vj) in enumerate(unique_edges):
        vi, vj = int(vi), int(vj)
        vertex_to_edges.setdefault(vi, []).append(idx)
        vertex_to_edges.setdefault(vj, []).append(idx)

    comp = _seam_component_labels(mask, unique_edges)

    endpoint_verts: set = set()
    for idx in np.where(mask)[0]:
        for v in (int(unique_edges[idx, 0]), int(unique_edges[idx, 1])):
            if sum(1 for e in vertex_to_edges.get(v, []) if mask[e]) == 1:
                endpoint_verts.add(v)

    for start_v in list(endpoint_verts):
        start_edges = [e for e in vertex_to_edges.get(start_v, []) if mask[e]]
        if not start_edges:
            continue
        start_comp = comp[start_edges[0]]

        path: list = []
        current_v = start_v
        visited_verts = {start_v}

        for _ in range(max_gap):
            candidates = [
                e for e in vertex_to_edges.get(current_v, [])
                if not mask[e] and e not in path
            ]
            if not candidates:
                break
            candidates.sort(key=lambda e: -probs[e])
            best_edge = candidates[0]
            path.append(best_edge)

            vi, vj = int(unique_edges[best_edge, 0]), int(unique_edges[best_edge, 1])
            next_v = vj if vi == current_v else vi
            if next_v in visited_verts:
                break
            visited_verts.add(next_v)

            next_seam = [e for e in vertex_to_edges.get(next_v, []) if mask[e]]
            if next_seam and comp[next_seam[0]] != start_comp:
                for e in path:
                    mask[e] = True
                break

            current_v = next_v

    return mask


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='UV seam inference worker.')
    parser.add_argument('data_npz', help='Path to mesh.npz with raw geometry arrays')
    parser.add_argument('weights_pth', help='Path to best_model.pth')
    parser.add_argument('threshold', type=float, help='Sigmoid threshold (0-1)')
    parser.add_argument('output_txt', help='Path to write seam edge indices')
    parser.add_argument('--model-type', default='graphsage', choices=['graphsage', 'gatv2'],
                        help='Model architecture (default: graphsage)')
    parser.add_argument('--min-component', type=int, default=3,
                        help='Min seam component size to keep (default: 3)')
    parser.add_argument('--max-gap', type=int, default=3,
                        help='Max gap edges to stitch (default: 3)')
    args = parser.parse_args()

    npz = np.load(args.data_npz)
    vertices = npz['vertices'].astype(np.float64)
    normals = npz['normals'].astype(np.float64)
    faces = npz['faces'].astype(np.int64)
    unique_edges = npz['unique_edges'].astype(np.int64)

    print(
        f'[UV Seam GNN] mesh: {len(vertices)} verts, '
        f'{len(faces)} faces, {len(unique_edges)} edges'
    )

    print('[UV Seam GNN] computing edge features...')
    features = compute_edge_features(vertices, normals, faces, unique_edges)  # [E, 11]

    print('[UV Seam GNN] building dual graph...')
    edge_index = build_dual_edge_index(faces, unique_edges)  # [2, D]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.from_numpy(features).float().to(device)
    edge_index = edge_index.to(device)

    if args.model_type == 'graphsage':
        model = DualGraphSAGE().to(device)
    else:
        model = DualGATv2().to(device)

    state = torch.load(args.weights_pth, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f'[UV Seam GNN] running {args.model_type} inference...')
    with torch.no_grad():
        logits = model(x, edge_index)
    probs = torch.sigmoid(logits).cpu().numpy()

    print('[UV Seam GNN] post-processing...')
    mask = threshold_and_clean(probs, unique_edges, args.threshold, args.min_component)
    mask = stitch_seam_gaps(probs, mask, unique_edges, args.max_gap)

    seam_indices = np.where(mask)[0].tolist()

    with open(args.output_txt, 'w') as f:
        f.write('\n'.join(map(str, seam_indices)))

    print(f'[UV Seam GNN] {len(seam_indices)} seam edges out of {len(unique_edges)} total.')


if __name__ == '__main__':
    main()
