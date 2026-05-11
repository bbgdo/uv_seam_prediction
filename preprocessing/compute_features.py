import sys
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

import trimesh

try:
    from preprocessing._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from preprocessing.canonical_mesh import resolve_endpoint_order  # noqa: E402
from preprocessing.topology import canonical_edge_key  # noqa: E402
from preprocessing.feature_registry import (  # noqa: E402
    ALL_ATOMIC_FEATURE_NAMES,
    DENSITY_CONFIG,
    PAPER14_FEATURE_NAMES,
    ResolvedFeatureSet,
    resolve_feature_selection,
)

ENDPOINT_ORDERS = ('fixed', 'random')
NORMALIZE_EPS = 1e-8
AO_RAY_COUNT = 32
ZSCORE_CLIP_RANGE = 3.0
DENSITY_EPS = DENSITY_CONFIG['eps']
DENSITY_LOG_CLIP = DENSITY_CONFIG['density_log_clip']


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norms < NORMALIZE_EPS, NORMALIZE_EPS, norms)


def build_edge_topology(mesh: trimesh.Trimesh) -> tuple[np.ndarray, dict]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edge_to_faces: dict[tuple, list] = {}

    for f_idx, face in enumerate(faces):
        for k in range(3):
            vi, vj = int(face[k]), int(face[(k + 1) % 3])
            key = canonical_edge_key(vi, vj)
            edge_to_faces.setdefault(key, []).append(f_idx)

    unique_edges = np.array(sorted(edge_to_faces.keys()), dtype=np.int64)
    return unique_edges, edge_to_faces




def compute_signed_dihedral(
    mesh: trimesh.Trimesh,
    unique_edges: np.ndarray,
    edge_to_faces: dict,
) -> np.ndarray:
    face_normals = mesh.face_normals
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    angles = np.zeros(len(unique_edges), dtype=np.float32)

    for idx, (vi, vj) in enumerate(unique_edges):
        key = canonical_edge_key(int(vi), int(vj))
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

    return (angles / np.pi).astype(np.float32)




def compute_vertex_gaussian_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_verts = len(vertices)

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]

    def _corner_angles(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
        na = np.linalg.norm(ea, axis=1)
        nb = np.linalg.norm(eb, axis=1)
        cos_a = np.einsum('ij,ij->i', ea, eb) / (na * nb + 1e-12)
        return np.arccos(np.clip(cos_a, -1.0, 1.0))

    angles_v0 = _corner_angles(v1 - v0, v2 - v0)
    angles_v1 = _corner_angles(v0 - v1, v2 - v1)
    angles_v2 = _corner_angles(v0 - v2, v1 - v2)

    angle_sum = (
        np.bincount(faces[:, 0], weights=angles_v0, minlength=n_verts)
        + np.bincount(faces[:, 1], weights=angles_v1, minlength=n_verts)
        + np.bincount(faces[:, 2], weights=angles_v2, minlength=n_verts)
    )

    edges_all = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], axis=0)
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


def _zscore_clip_normalize(values: np.ndarray) -> np.ndarray:
    mean = values.mean()
    std = values.std() + 1e-8
    z = (values - mean) / std
    z = np.clip(z, -ZSCORE_CLIP_RANGE, ZSCORE_CLIP_RANGE)
    return (z / ZSCORE_CLIP_RANGE).astype(np.float32)




def _generate_hemisphere_samples(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    samples = np.zeros((n_samples, 3), dtype=np.float64)
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n_samples):
        theta = np.arccos(1 - (i + 0.5) / n_samples)
        phi = 2 * np.pi * i / golden_ratio
        samples[i] = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]

    return samples


def _rotation_matrix_to_align(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    from_vec = from_vec / (np.linalg.norm(from_vec) + 1e-12)
    to_vec = to_vec / (np.linalg.norm(to_vec) + 1e-12)

    cross = np.cross(from_vec, to_vec)
    dot = np.dot(from_vec, to_vec)

    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
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


def _build_ray_intersector(mesh: trimesh.Trimesh, test_origin: np.ndarray, test_direction: np.ndarray):
    for loader in [
        lambda: __import__('trimesh.ray.ray_pyembree', fromlist=['RayMeshIntersector']).RayMeshIntersector,
        lambda: __import__('trimesh.ray.ray_triangle', fromlist=['RayMeshIntersector']).RayMeshIntersector,
    ]:
        try:
            cls = loader()
            candidate = cls(mesh)
            candidate.intersects_any(test_origin, test_direction)
            return candidate
        except Exception:
            continue
    return None


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.zeros(3, dtype=np.float64)
    ref[int(np.argmin(np.abs(direction)))] = 1.0

    t1 = np.cross(direction, ref)
    if np.linalg.norm(t1) < 1e-12:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        t1 = np.cross(direction, ref)
    t1 = _safe_normalize(t1[None, :])[0]
    t2 = _safe_normalize(np.cross(direction, t1)[None, :])[0]
    return t1, t2


def compute_vertex_ao(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    n_verts = len(vertices)

    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    epsilon = 1e-4 * bbox_diag

    rng = np.random.default_rng(42)
    hemisphere_samples = _generate_hemisphere_samples(AO_RAY_COUNT, rng)
    z_axis = np.array([0.0, 0.0, 1.0])

    intersector = None
    for loader in [
        lambda: __import__('trimesh.ray.ray_pyembree', fromlist=['RayMeshIntersector']).RayMeshIntersector,
        lambda: __import__('trimesh.ray.ray_triangle', fromlist=['RayMeshIntersector']).RayMeshIntersector,
    ]:
        try:
            cls = loader()
            candidate = cls(mesh)
            test_origin = vertices[0:1] + normals[0:1] * epsilon
            test_dir = normals[0:1]
            candidate.intersects_any(test_origin, test_dir)
            intersector = candidate
            break
        except Exception:
            continue

    if intersector is None:
        raise RuntimeError(
            'AO raycasting requires pyembree or trimesh ray_triangle; '
            'install pyembree or ensure trimesh ray_triangle is functional.'
        )

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

        try:
            hits = intersector.intersects_any(batch_origins, batch_directions)
        except Exception as exc:
            raise RuntimeError(f'AO raycasting failed at batch {batch_start}: {exc}') from exc
        hits = hits.reshape(batch_end - batch_start, AO_RAY_COUNT)
        ao_values[batch_start:batch_end] = hits.mean(axis=1)

    return ao_values




def compute_edge_sdf(
    mesh: trimesh.Trimesh,
    unique_edges: np.ndarray,
    edge_to_faces: dict,
) -> np.ndarray:
    n_edges = len(unique_edges)
    if n_edges == 0:
        return np.zeros(0, dtype=np.float32)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    if len(vertices) == 0:
        return np.ones(n_edges, dtype=np.float32)

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    bbox_diag = float(np.linalg.norm(bounds[1] - bounds[0])) if bounds.shape == (2, 3) else 0.0
    if not np.isfinite(bbox_diag) or bbox_diag < 1e-12:
        bbox_diag = 1e-12
    epsilon = 1e-4 * bbox_diag
    min_hit_distance = max(2.0 * epsilon, 1e-12)

    edge_vi = unique_edges[:, 0].astype(np.int64)
    edge_vj = unique_edges[:, 1].astype(np.int64)
    midpoints = 0.5 * (vertices[edge_vi] + vertices[edge_vj])
    edge_lengths = np.linalg.norm(vertices[edge_vj] - vertices[edge_vi], axis=1)

    base_normals = np.zeros((n_edges, 3), dtype=np.float64)
    source_face_sets: list[set[int]] = []
    for edge_idx, (vi, vj) in enumerate(unique_edges):
        key = canonical_edge_key(int(vi), int(vj))
        faces = [
            int(face_idx)
            for face_idx in edge_to_faces.get(key, [])
            if 0 <= int(face_idx) < len(face_normals)
        ]
        source_face_sets.append(set(faces))
        if len(faces) >= 2:
            base_normals[edge_idx] = face_normals[faces[0]] + face_normals[faces[1]]
        elif len(faces) == 1:
            base_normals[edge_idx] = face_normals[faces[0]]

    normal_lengths = np.linalg.norm(base_normals, axis=1)
    valid_edges = (normal_lengths > 1e-8) & (edge_lengths > 1e-12 * bbox_diag)
    if not np.any(valid_edges):
        return np.ones(n_edges, dtype=np.float32)

    directions = -_safe_normalize(base_normals)
    directions[~valid_edges] = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    cone_angle = np.deg2rad(17.5)
    cos_a = float(np.cos(cone_angle))
    sin_a = float(np.sin(cone_angle))
    ray_directions = np.empty((n_edges, 5, 3), dtype=np.float64)
    for edge_idx, direction in enumerate(directions):
        if not valid_edges[edge_idx]:
            ray_directions[edge_idx] = direction
            continue

        t1, t2 = _orthonormal_basis(direction)
        rays = np.array([
            direction,
            cos_a * direction + sin_a * t1,
            cos_a * direction - sin_a * t1,
            cos_a * direction + sin_a * t2,
            cos_a * direction - sin_a * t2,
        ], dtype=np.float64)
        ray_directions[edge_idx] = _safe_normalize(rays)

    origins = midpoints + epsilon * directions
    flat_origins = np.repeat(origins, 5, axis=0)
    flat_directions = ray_directions.reshape((-1, 3))
    valid_ray_mask = np.repeat(valid_edges, 5)
    first_valid_ray = int(np.flatnonzero(valid_ray_mask)[0])

    intersector = _build_ray_intersector(
        mesh,
        flat_origins[first_valid_ray:first_valid_ray + 1],
        flat_directions[first_valid_ray:first_valid_ray + 1],
    )
    if intersector is None:
        raise RuntimeError(
            'thickness_sdf raycasting requires pyembree or trimesh ray_triangle; '
            'install pyembree or ensure trimesh ray_triangle is functional.'
        )

    try:
        hit_faces, hit_rays, hit_locations = intersector.intersects_id(
            flat_origins,
            flat_directions,
            multiple_hits=True,
            return_locations=True,
        )
    except Exception as exc:
        raise RuntimeError(f'thickness_sdf raycasting failed: {exc}') from exc

    nearest_by_ray = np.full(len(flat_origins), np.inf, dtype=np.float64)
    for face_idx, ray_idx, location in zip(hit_faces, hit_rays, hit_locations):
        ray_idx = int(ray_idx)
        if ray_idx < 0 or ray_idx >= len(flat_origins) or not valid_ray_mask[ray_idx]:
            continue

        edge_idx = ray_idx // 5
        if int(face_idx) in source_face_sets[edge_idx]:
            continue

        distance = float(np.linalg.norm(np.asarray(location, dtype=np.float64) - flat_origins[ray_idx]))
        if not np.isfinite(distance) or distance <= min_hit_distance:
            continue
        if distance < nearest_by_ray[ray_idx]:
            nearest_by_ray[ray_idx] = distance

    distances = np.full(n_edges, bbox_diag, dtype=np.float64)
    nearest_by_edge = nearest_by_ray.reshape((n_edges, 5))
    for edge_idx in np.flatnonzero(valid_edges):
        valid_distances = nearest_by_edge[edge_idx][np.isfinite(nearest_by_edge[edge_idx])]
        if len(valid_distances) > 0:
            distances[edge_idx] = float(np.median(valid_distances))

    thickness = np.clip(distances / bbox_diag, 0.0, 1.0)
    thickness[~np.isfinite(thickness)] = 1.0
    return thickness.astype(np.float32)


def detect_symmetry_axis(
    mesh: trimesh.Trimesh, threshold_ratio: float = 0.8
) -> int | None:
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




def compute_symmetry_distance(
    mesh: trimesh.Trimesh, unique_edges: np.ndarray
) -> np.ndarray:
    axis = detect_symmetry_axis(mesh)
    if axis is None:
        return np.full(len(unique_edges), 0.5, dtype=np.float32)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    midpoints = (verts[unique_edges[:, 0]] + verts[unique_edges[:, 1]]) / 2.0
    max_extent = np.abs(verts[:, axis]).max() + 1e-8
    distances = np.abs(midpoints[:, axis]) / max_extent
    return distances.astype(np.float32)


def compute_vertex_support_area(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_verts = len(mesh.vertices)
    support = np.zeros(n_verts, dtype=np.float64)
    if len(faces) == 0:
        return support

    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    weights = np.repeat(face_areas / 3.0, 3)
    np.add.at(support, faces.reshape(-1), weights)
    return support


def _build_vertex_adjacency(faces: np.ndarray, n_verts: int) -> list[set[int]]:
    adjacency = [set() for _ in range(n_verts)]
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    return adjacency


def _two_ring_neighborhood(adjacency: list[set[int]], vertex_idx: int) -> set[int]:
    neighborhood = set(adjacency[vertex_idx])
    for neighbor in adjacency[vertex_idx]:
        neighborhood.update(adjacency[neighbor])
    neighborhood.discard(vertex_idx)
    if not neighborhood:
        neighborhood.add(vertex_idx)
    return neighborhood


def compute_vertex_relative_density(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_verts = len(mesh.vertices)
    support_area = compute_vertex_support_area(mesh)
    local_scale = np.sqrt(support_area + DENSITY_EPS)
    adjacency = _build_vertex_adjacency(faces, n_verts)

    density = np.zeros(n_verts, dtype=np.float64)
    for vertex_idx in range(n_verts):
        neighborhood = _two_ring_neighborhood(adjacency, vertex_idx)
        median_scale = float(np.median(local_scale[list(neighborhood)]))
        density[vertex_idx] = np.log(median_scale + DENSITY_EPS) - np.log(local_scale[vertex_idx] + DENSITY_EPS)
    return density.astype(np.float32)


def _normalize_vertex_relative_density(vertex_density: np.ndarray) -> np.ndarray:
    clipped = np.clip(vertex_density, -DENSITY_LOG_CLIP, DENSITY_LOG_CLIP)
    return (clipped / DENSITY_LOG_CLIP).astype(np.float32)


def compute_edge_relative_density(
    mesh: trimesh.Trimesh,
    unique_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertex_density_raw = compute_vertex_relative_density(mesh)
    vertex_density = _normalize_vertex_relative_density(vertex_density_raw)
    vi = unique_edges[:, 0]
    vj = unique_edges[:, 1]
    density_i = vertex_density[vi]
    density_j = vertex_density[vj]
    density_mean = ((density_i + density_j) * 0.5).astype(np.float32)
    density_diff = (0.5 * np.abs(density_i - density_j)).astype(np.float32)
    return density_mean, density_diff


def _normalized_vertex_basics(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64).astype(np.float32)

    com = mesh.center_mass if bool(getattr(mesh, 'is_volume', False)) else verts.mean(axis=0)
    if not np.all(np.isfinite(com)):
        com = verts.mean(axis=0)
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) + 1e-8
    pos_norm = ((verts - com) / bbox_diag).astype(np.float32)

    gauss_curv = compute_vertex_gaussian_curvature(mesh)
    gauss_curv_norm = _zscore_clip_normalize(gauss_curv)
    return pos_norm, normals, gauss_curv_norm


def _ordered_endpoint_features(
    vertex_features: np.ndarray,
    unique_edges: np.ndarray,
    endpoint_order: str,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if endpoint_order not in ENDPOINT_ORDERS:
        raise ValueError(f"endpoint_order must be one of {ENDPOINT_ORDERS}, got: {endpoint_order}")

    vi = unique_edges[:, 0].copy()
    vj = unique_edges[:, 1].copy()
    if endpoint_order == 'random':
        rng = np.random.default_rng(rng_seed)
        swap = rng.random(len(unique_edges)) < 0.5
        vi[swap], vj[swap] = vj[swap], vi[swap]

    return vertex_features[vi], vertex_features[vj]


def _compute_atomic_edge_columns(
    mesh: trimesh.Trimesh,
    unique_edges: np.ndarray,
    edge_to_faces: dict,
    feature_names: tuple[str, ...],
    endpoint_order: str,
    rng_seed: int,
) -> dict[str, np.ndarray]:
    unknown = sorted(set(feature_names) - set(ALL_ATOMIC_FEATURE_NAMES))
    if unknown:
        raise ValueError(f"unknown atomic feature name(s): {unknown}")

    columns: dict[str, np.ndarray] = {}
    pos_norm, normals_f32, gauss_curv_norm = _normalized_vertex_basics(mesh)
    base_vertex_features = np.concatenate([
        pos_norm,
        normals_f32,
        gauss_curv_norm[:, None],
    ], axis=1).astype(np.float32)
    vi_base, vj_base = _ordered_endpoint_features(base_vertex_features, unique_edges, endpoint_order, rng_seed)

    base_i_names = PAPER14_FEATURE_NAMES[:7]
    base_j_names = PAPER14_FEATURE_NAMES[7:]
    for col_idx, name in enumerate(base_i_names):
        columns[name] = vi_base[:, col_idx].astype(np.float32)
    for col_idx, name in enumerate(base_j_names):
        columns[name] = vj_base[:, col_idx].astype(np.float32)

    if 'ao_i' in feature_names or 'ao_j' in feature_names:
        ao = compute_vertex_ao(mesh)[:, None]
        vi_ao, vj_ao = _ordered_endpoint_features(ao, unique_edges, endpoint_order, rng_seed)
        columns['ao_i'] = vi_ao[:, 0].astype(np.float32)
        columns['ao_j'] = vj_ao[:, 0].astype(np.float32)

    if 'signed_dihedral' in feature_names:
        columns['signed_dihedral'] = compute_signed_dihedral(mesh, unique_edges, edge_to_faces)

    if 'symmetry_dist' in feature_names:
        columns['symmetry_dist'] = compute_symmetry_distance(mesh, unique_edges)

    if 'density_mean' in feature_names or 'density_diff' in feature_names:
        density_mean, density_diff = compute_edge_relative_density(mesh, unique_edges)
        columns['density_mean'] = density_mean
        columns['density_diff'] = density_diff

    if 'thickness_sdf' in feature_names:
        columns['thickness_sdf'] = compute_edge_sdf(mesh, unique_edges, edge_to_faces)

    return columns


def compute_edge_features_for_selection(
    mesh: trimesh.Trimesh,
    selection: ResolvedFeatureSet,
    endpoint_order: str = 'auto',
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    endpoint_order = resolve_endpoint_order(selection.feature_group, endpoint_order)
    unique_edges, edge_to_faces = build_edge_topology(mesh)
    columns = _compute_atomic_edge_columns(
        mesh,
        unique_edges,
        edge_to_faces,
        selection.feature_names,
        endpoint_order,
        rng_seed,
    )
    features = np.stack([columns[name] for name in selection.feature_names], axis=1).astype(np.float32)
    return features, unique_edges, edge_to_faces


def compute_edge_features(
    mesh: trimesh.Trimesh,
    feature_group: str = 'paper14',
    endpoint_order: str = 'auto',
    rng_seed: int = 42,
    *,
    enable_ao: bool = False,
    enable_dihedral: bool = False,
    enable_symmetry: bool = False,
    enable_density: bool = False,
    enable_thickness_sdf: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    selection = resolve_feature_selection(
        feature_group,
        enable_ao=enable_ao,
        enable_dihedral=enable_dihedral,
        enable_symmetry=enable_symmetry,
        enable_density=enable_density,
        enable_thickness_sdf=enable_thickness_sdf,
    )
    return compute_edge_features_for_selection(mesh, selection, endpoint_order=endpoint_order, rng_seed=rng_seed)


FEATURE_NAMES = PAPER14_FEATURE_NAMES


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
