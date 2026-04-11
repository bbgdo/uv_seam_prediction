import sys
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import trimesh  # noqa: E402


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norms < eps, eps, norms)


def build_edge_topology(mesh: trimesh.Trimesh) -> tuple[np.ndarray, dict]:
    """Build sorted unique edges and edge-to-faces mapping.

    Returns:
        unique_edges: [E, 2] int64, sorted with vi < vj
        edge_to_faces: dict mapping (vi, vj) -> [face_idx, ...]
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edge_to_faces: dict[tuple, list] = {}

    for f_idx, face in enumerate(faces):
        for k in range(3):
            vi, vj = int(face[k]), int(face[(k + 1) % 3])
            key = (min(vi, vj), max(vi, vj))
            edge_to_faces.setdefault(key, []).append(f_idx)

    unique_edges = np.array(sorted(edge_to_faces.keys()), dtype=np.int64)
    return unique_edges, edge_to_faces


def compute_edge_length(mesh: trimesh.Trimesh, unique_edges: np.ndarray) -> np.ndarray:
    """Feature 0: edge length, normalized per-mesh by max edge length."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    lengths = np.linalg.norm(verts[unique_edges[:, 1]] - verts[unique_edges[:, 0]], axis=1)
    max_len = lengths.max() + 1e-8
    return (lengths / max_len).astype(np.float32)


def compute_signed_dihedral(
    mesh: trimesh.Trimesh,
    unique_edges: np.ndarray,
    edge_to_faces: dict,
) -> np.ndarray:
    """Feature 1: signed dihedral angle, normalized to [-1, 1].
    Positive = convex, negative = concave. Boundary edges = 0."""
    face_normals = mesh.face_normals
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    angles = np.zeros(len(unique_edges), dtype=np.float32)

    for idx, (vi, vj) in enumerate(unique_edges):
        key = (min(vi, vj), max(vi, vj))
        face_list = edge_to_faces.get(key, [])
        if len(face_list) < 2:
            continue

        n0 = face_normals[face_list[0]].astype(np.float64)
        n1 = face_normals[face_list[1]].astype(np.float64)

        cos_a = np.clip(np.dot(n0, n1), -1.0, 1.0)
        unsigned_angle = np.arccos(cos_a)

        edge_dir = vertices[vj] - vertices[vi]
        edge_norm = np.linalg.norm(edge_dir)
        if edge_norm > 1e-8:
            edge_dir /= edge_norm
            cross = np.cross(n0, n1)
            angles[idx] = float(unsigned_angle * np.sign(np.dot(cross, edge_dir) + 1e-12))
        else:
            angles[idx] = float(unsigned_angle)

    # normalize to [-1, 1]
    return (angles / np.pi).astype(np.float32)


def compute_sharpness(signed_dihedral_normalized: np.ndarray) -> np.ndarray:
    """Feature 2: sharpness = |signed_dihedral / pi|. 0 = flat, 1 = knife-edge.
    Input is already normalized to [-1, 1], so sharpness = abs(input)."""
    return np.abs(signed_dihedral_normalized).astype(np.float32)


def compute_delta_normal(mesh: trimesh.Trimesh, unique_edges: np.ndarray) -> np.ndarray:
    """Feature 4: magnitude of vertex normal difference, normalized to [0, 1]."""
    vn = np.asarray(mesh.vertex_normals, dtype=np.float32)
    delta = np.linalg.norm(vn[unique_edges[:, 0]] - vn[unique_edges[:, 1]], axis=1)
    # max possible magnitude of difference between unit vectors is 2
    return (delta / 2.0).astype(np.float32)


def compute_dot_normal(mesh: trimesh.Trimesh, unique_edges: np.ndarray) -> np.ndarray:
    """Feature 5: dot product of normalized vertex normals. Range [-1, 1]."""
    vn = np.asarray(mesh.vertex_normals, dtype=np.float32)
    n_vi = _safe_normalize(vn[unique_edges[:, 0]])
    n_vj = _safe_normalize(vn[unique_edges[:, 1]])
    return np.einsum('ij,ij->i', n_vi, n_vj).astype(np.float32)


def compute_vertex_gaussian_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """Discrete Gaussian curvature via angle defect method (vectorized).

    K_v = 2pi - sum(incident face angles at v) for interior vertices
    K_v = pi - sum(incident face angles at v) for boundary vertices
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_verts = len(vertices)

    # vectorized angle computation: one entry per (face, corner)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]

    def _corner_angles(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
        na = np.linalg.norm(ea, axis=1)
        nb = np.linalg.norm(eb, axis=1)
        cos_a = np.einsum('ij,ij->i', ea, eb) / (na * nb + 1e-12)
        return np.arccos(np.clip(cos_a, -1.0, 1.0))

    angles_v0 = _corner_angles(v1 - v0, v2 - v0)  # [F]
    angles_v1 = _corner_angles(v0 - v1, v2 - v1)  # [F]
    angles_v2 = _corner_angles(v0 - v2, v1 - v2)  # [F]

    angle_sum = (
        np.bincount(faces[:, 0], weights=angles_v0, minlength=n_verts)
        + np.bincount(faces[:, 1], weights=angles_v1, minlength=n_verts)
        + np.bincount(faces[:, 2], weights=angles_v2, minlength=n_verts)
    )

    # detect boundary vertices via edge face count
    edges_all = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], axis=0)  # [3F, 2]
    edges_sorted = np.sort(edges_all, axis=1)
    encoded = edges_sorted[:, 0] * n_verts + edges_sorted[:, 1]
    unique_enc, counts = np.unique(encoded, return_counts=True)
    boundary_enc = unique_enc[counts == 1]
    boundary_vi = (boundary_enc // n_verts).astype(np.int64)
    boundary_vj = (boundary_enc % n_verts).astype(np.int64)
    is_boundary = np.zeros(n_verts, dtype=bool)
    is_boundary[boundary_vi] = True
    is_boundary[boundary_vj] = True

    curvatures = np.where(is_boundary, np.pi - angle_sum, 2.0 * np.pi - angle_sum)
    return curvatures.astype(np.float32)


def _zscore_clip_normalize(values: np.ndarray, clip_range: float = 3.0) -> np.ndarray:
    mean = values.mean()
    std = values.std() + 1e-8
    z = (values - mean) / std
    z = np.clip(z, -clip_range, clip_range)
    return (z / clip_range).astype(np.float32)


def compute_gauss_curvature_features(
    mesh: trimesh.Trimesh, unique_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Features 6-7: mean and diff of Gaussian curvature at edge endpoints."""
    curvatures = compute_vertex_gaussian_curvature(mesh)
    curvatures_norm = _zscore_clip_normalize(curvatures)

    k_vi = curvatures_norm[unique_edges[:, 0]]
    k_vj = curvatures_norm[unique_edges[:, 1]]

    gauss_mean = ((k_vi + k_vj) / 2.0).astype(np.float32)
    gauss_diff = np.abs(k_vi - k_vj).astype(np.float32)
    return gauss_mean, gauss_diff


def _generate_hemisphere_samples(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate uniformly distributed points on the upper hemisphere (z >= 0)."""
    # Fibonacci hemisphere sampling for deterministic, well-distributed points
    samples = np.zeros((n_samples, 3), dtype=np.float64)
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n_samples):
        theta = np.arccos(1 - (i + 0.5) / n_samples)  # [0, pi/2] for hemisphere
        phi = 2 * np.pi * i / golden_ratio
        samples[i] = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]

    return samples


def _rotation_matrix_to_align(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """Rotation matrix that aligns from_vec to to_vec using Rodrigues' formula."""
    from_vec = from_vec / (np.linalg.norm(from_vec) + 1e-12)
    to_vec = to_vec / (np.linalg.norm(to_vec) + 1e-12)

    cross = np.cross(from_vec, to_vec)
    dot = np.dot(from_vec, to_vec)

    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        # 180-degree rotation — find perpendicular axis
        perp = np.array([1, 0, 0]) if abs(from_vec[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, from_vec) * from_vec
        perp /= np.linalg.norm(perp) + 1e-12
        return 2 * np.outer(perp, perp) - np.eye(3)

    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])
    return np.eye(3) + skew + skew @ skew / (1 + dot)


def compute_vertex_ao(mesh: trimesh.Trimesh, n_rays: int = 32) -> np.ndarray:
    """Ambient occlusion per vertex via raycasting.

    AO = fraction of hemisphere rays that hit other geometry.
    Falls back to a normal-based approximation if raycasting fails.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    n_verts = len(vertices)

    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    epsilon = 1e-4 * bbox_diag

    rng = np.random.default_rng(42)
    hemisphere_samples = _generate_hemisphere_samples(n_rays, rng)
    z_axis = np.array([0.0, 0.0, 1.0])

    # try ray intersection — validate with a test ray before committing
    intersector = None
    for loader in [
        lambda: __import__('trimesh.ray.ray_pyembree', fromlist=['RayMeshIntersector']).RayMeshIntersector,
        lambda: __import__('trimesh.ray.ray_triangle', fromlist=['RayMeshIntersector']).RayMeshIntersector,
    ]:
        try:
            cls = loader()
            candidate = cls(mesh)
            # smoke test: single ray to verify dependencies are working
            test_origin = vertices[0:1] + normals[0:1] * epsilon
            test_dir = normals[0:1]
            candidate.intersects_any(test_origin, test_dir)
            intersector = candidate
            break
        except Exception:
            continue

    if intersector is not None:
        ao_values = np.zeros(n_verts, dtype=np.float32)
        batch_size = 256
        for batch_start in range(0, n_verts, batch_size):
            batch_end = min(batch_start + batch_size, n_verts)
            batch_origins = []
            batch_directions = []

            for v_idx in range(batch_start, batch_end):
                normal = normals[v_idx]
                origin = vertices[v_idx] + normal * epsilon
                rot = _rotation_matrix_to_align(z_axis, normal)
                directions = (rot @ hemisphere_samples.T).T

                for d in directions:
                    batch_origins.append(origin)
                    batch_directions.append(d)

            batch_origins = np.array(batch_origins)
            batch_directions = np.array(batch_directions)

            hits = intersector.intersects_any(batch_origins, batch_directions)
            hits = hits.reshape(batch_end - batch_start, n_rays)
            ao_values[batch_start:batch_end] = hits.mean(axis=1)

        return ao_values

    # fallback: normal-based AO approximation
    print('  [ao] raycasting unavailable, using normal-based approximation')
    return _ao_normal_approximation(mesh)


def _ao_normal_approximation(mesh: trimesh.Trimesh) -> np.ndarray:
    """Crude AO proxy: for each vertex, check how many nearby vertices
    have normals pointing toward it (suggesting occlusion).

    Uses k-NN (k=16) instead of radius search for fully vectorized computation.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    n_verts = len(vertices)

    if cKDTree is None:
        return np.full(n_verts, 0.5, dtype=np.float32)

    k = min(16, n_verts - 1)
    if k < 1:
        return np.full(n_verts, 0.5, dtype=np.float32)

    tree = cKDTree(vertices)
    # single vectorized batch query — avoids O(N) Python loop
    _, indices = tree.query(vertices, k=k + 1)  # [N, k+1], col 0 is self
    nb_idx = indices[:, 1:]  # [N, k]

    # directions from each neighbor to each vertex: [N, k, 3]
    dirs = vertices[:, None, :] - vertices[nb_idx]
    dirs = dirs / (np.linalg.norm(dirs, axis=2, keepdims=True) + 1e-12)

    # alignment of neighbor normals with direction toward vertex
    n_nbr = normals[nb_idx]  # [N, k, 3]
    alignment = np.einsum('nki,nki->nk', n_nbr, dirs)  # [N, k]
    return np.clip(np.mean(np.maximum(alignment, 0), axis=1), 0, 1).astype(np.float32)


def compute_ao_features(
    mesh: trimesh.Trimesh, unique_edges: np.ndarray, n_rays: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Features 8-9: mean and diff of ambient occlusion at edge endpoints."""
    ao = compute_vertex_ao(mesh, n_rays=n_rays)
    ao_vi = ao[unique_edges[:, 0]]
    ao_vj = ao[unique_edges[:, 1]]

    ao_mean = ((ao_vi + ao_vj) / 2.0).astype(np.float32)
    ao_diff = np.abs(ao_vi - ao_vj).astype(np.float32)
    return ao_mean, ao_diff


def detect_symmetry_axis(
    mesh: trimesh.Trimesh, threshold_ratio: float = 0.8
) -> int | None:
    """Detect dominant mirror symmetry axis (0=X, 1=Y, 2=Z).
    Returns None if no axis has a match ratio above threshold."""
    if cKDTree is None:
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    match_tol = bbox_diag * 0.01

    tree = cKDTree(vertices)
    best_axis = None
    best_ratio = 0.0

    for axis in range(3):
        reflected = vertices.copy()
        reflected[:, axis] *= -1
        dists, _ = tree.query(reflected)
        ratio = np.mean(dists < match_tol)
        if ratio > best_ratio:
            best_ratio = ratio
            best_axis = axis

    if best_ratio >= threshold_ratio:
        return best_axis
    return None


def compute_centroid_position(
    mesh: trimesh.Trimesh,
    unique_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge midpoint position, centered at mesh COM and scaled by bbox diagonal.

    COM-centering + single-scalar scaling preserves proportions across meshes
    (unlike per-axis bbox normalization which distorts aspect ratios).
    Humanoid meshes from the same generator get consistent normalized coords:
    head ≈ top, feet ≈ bottom, arms ≈ sides.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    midpoints = (verts[unique_edges[:, 0]] + verts[unique_edges[:, 1]]) / 2.0

    com = mesh.center_mass if hasattr(mesh, 'center_mass') else verts.mean(axis=0)
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) + 1e-8

    centered = (midpoints - com) / bbox_diag  # roughly [-0.5, 0.5]
    return (
        centered[:, 0].astype(np.float32),
        centered[:, 1].astype(np.float32),
        centered[:, 2].astype(np.float32),
    )


def compute_edge_normal(
    mesh: trimesh.Trimesh,
    unique_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average vertex normal at edge midpoint, normalized to unit length.

    Encodes which direction the surface faces at each edge.
    Front-facing edges have high Z, side-facing edges have high |X|, etc.
    """
    vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
    avg = (vn[unique_edges[:, 0]] + vn[unique_edges[:, 1]]) / 2.0
    avg = _safe_normalize(avg)
    return (
        avg[:, 0].astype(np.float32),
        avg[:, 1].astype(np.float32),
        avg[:, 2].astype(np.float32),
    )


def compute_symmetry_distance(
    mesh: trimesh.Trimesh, unique_edges: np.ndarray
) -> np.ndarray:
    """Feature 10: distance from edge midpoint to nearest symmetry plane, normalized to [0, 1]."""
    axis = detect_symmetry_axis(mesh)
    if axis is None:
        return np.full(len(unique_edges), 0.5, dtype=np.float32)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    midpoints = (verts[unique_edges[:, 0]] + verts[unique_edges[:, 1]]) / 2.0
    max_extent = np.abs(verts[:, axis]).max() + 1e-8
    distances = np.abs(midpoints[:, axis]) / max_extent
    return distances.astype(np.float32)


def compute_edge_features(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute all 16 edge features for a mesh.

    Returns:
        edge_features: [E, 16] float32 array
        unique_edges: [E, 2] int64 array (vi < vj)
        edge_to_faces: dict mapping (vi, vj) -> [face_idx, ...]
    """
    unique_edges, edge_to_faces = build_edge_topology(mesh)

    f0_length = compute_edge_length(mesh, unique_edges)
    f1_dihedral = compute_signed_dihedral(mesh, unique_edges, edge_to_faces)
    f2_sharpness = compute_sharpness(f1_dihedral)
    f3_delta_n = compute_delta_normal(mesh, unique_edges)
    f4_dot_n = compute_dot_normal(mesh, unique_edges)
    f5_gauss_mean, f6_gauss_diff = compute_gauss_curvature_features(mesh, unique_edges)
    f7_ao_mean, f8_ao_diff = compute_ao_features(mesh, unique_edges)
    f9_symmetry = compute_symmetry_distance(mesh, unique_edges)
    f10_cx, f11_cy, f12_cz = compute_centroid_position(mesh, unique_edges)
    f13_nx, f14_ny, f15_nz = compute_edge_normal(mesh, unique_edges)

    features = np.stack([
        f0_length, f1_dihedral, f2_sharpness,
        f3_delta_n, f4_dot_n, f5_gauss_mean, f6_gauss_diff,
        f7_ao_mean, f8_ao_diff, f9_symmetry,
        f10_cx, f11_cy, f12_cz,
        f13_nx, f14_ny, f15_nz,
    ], axis=1).astype(np.float32)

    return features, unique_edges, edge_to_faces


FEATURE_NAMES = [
    'edge_length', 'signed_dihedral', 'sharpness',
    'delta_normal', 'dot_normal', 'gauss_curv_mean', 'gauss_curv_diff',
    'ao_mean', 'ao_diff', 'symmetry_dist',
    'centroid_x', 'centroid_y', 'centroid_z',
    'avg_normal_x', 'avg_normal_y', 'avg_normal_z',
]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compute_features.py <mesh.obj>")
        sys.exit(1)

    mesh_path = Path(sys.argv[1])
    if not mesh_path.exists():
        print(f"[error] file not found: {mesh_path}")
        sys.exit(1)

    mesh = trimesh.load(str(mesh_path), process=False, force='mesh')
    print(f"mesh: {mesh_path.name}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    features, edges, _ = compute_edge_features(mesh)
    print(f"edges: {len(edges)}, features: {features.shape[1]}\n")

    print(f"{'feature':<20s} {'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s} {'nan?':>6s} {'inf?':>6s}")
    print('-' * 72)
    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]
        has_nan = 'YES' if np.any(np.isnan(col)) else 'no'
        has_inf = 'YES' if np.any(np.isinf(col)) else 'no'
        print(f'{name:<20s} {col.min():>10.4f} {col.max():>10.4f} {col.mean():>10.4f} {col.std():>10.4f} {has_nan:>6s} {has_inf:>6s}')
