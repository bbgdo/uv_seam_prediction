import sys
from pathlib import Path

import numpy as np


def parse_obj_with_uv(path: str) -> dict:
    """Parse .obj file extracting vertices, faces, UV coords, and UV face indices.

    Handles triangulated meshes and simple quads (fan-triangulated).
    Face format supported: `f v`, `f v/vt`, `f v/vt/vn`, `f v//vn`.

    Returns:
        vertices:  [N, 3] float64
        faces:     [F, 3] int64  (0-indexed vertex indices)
        uv_coords: [M, 2] float64 | None
        uv_faces:  [F, 3] int64  | None  (0-indexed UV indices per corner)
    """
    vertices: list = []
    uv_coords: list = []
    faces: list = []
    uv_faces: list = []

    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            tok = parts[0]

            if tok == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif tok == 'vt':
                uv_coords.append([float(parts[1]), float(parts[2])])

            elif tok == 'f':
                face_v, face_vt = [], []
                for token in parts[1:]:
                    idx = token.split('/')
                    face_v.append(int(idx[0]) - 1)
                    if len(idx) > 1 and idx[1]:
                        face_vt.append(int(idx[1]) - 1)

                # fan-triangulate in case of quads/n-gons
                for i in range(1, len(face_v) - 1):
                    faces.append([face_v[0], face_v[i], face_v[i + 1]])
                    if len(face_vt) == len(face_v):
                        uv_faces.append([face_vt[0], face_vt[i], face_vt[i + 1]])

    result = {
        'vertices': np.array(vertices, dtype=np.float64),
        'faces': np.array(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64),
    }

    if uv_coords and uv_faces and len(uv_faces) == len(faces):
        result['uv_coords'] = np.array(uv_coords, dtype=np.float64)
        result['uv_faces'] = np.array(uv_faces, dtype=np.int64)
    else:
        result['uv_coords'] = None
        result['uv_faces'] = None

    return result


def _triangle_areas_3d(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _triangle_areas_uv(uv_coords: np.ndarray, uv_faces: np.ndarray) -> np.ndarray:
    u0 = uv_coords[uv_faces[:, 0]]
    u1 = uv_coords[uv_faces[:, 1]]
    u2 = uv_coords[uv_faces[:, 2]]
    # 2D cross product gives signed area * 2
    return 0.5 * ((u1[:, 0] - u0[:, 0]) * (u2[:, 1] - u0[:, 1])
                  - (u2[:, 0] - u0[:, 0]) * (u1[:, 1] - u0[:, 1]))


def _resolve_uvs(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray | None,
    uv_faces: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:

    if uv_faces is None:
        # assume uv_coords == vertex positions projected to 2D (XY), 1:1 mapping
        if uv_coords is None:
            raise ValueError('No UV data available.')
        return uv_coords, faces
    return uv_coords, uv_faces


def _jacobians(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    uv_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 2×2 Jacobian of the UV map on each triangle.

    Maps from the 2D local frame of the 3D triangle to 2D UV space.

    Returns (J, areas_3d) where J is [F, 2, 2] and areas_3d is [F].
    """
    v0 = vertices[faces[:, 0]]   # [F, 3]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # local 2D coordinate system on 3D triangle
    e1_3d = v1 - v0             # [F, 3]
    e2_3d = v2 - v0

    e1_len = np.linalg.norm(e1_3d, axis=1, keepdims=True).clip(min=1e-12)  # [F, 1]
    x_axis = e1_3d / e1_len     # [F, 3]

    # y_axis = component of e2 perpendicular to x_axis, then normalized
    proj = np.sum(e2_3d * x_axis, axis=1, keepdims=True) * x_axis
    y3d = e2_3d - proj
    y_len = np.linalg.norm(y3d, axis=1, keepdims=True).clip(min=1e-12)
    y_axis = y3d / y_len

    # local 2D coords of e1 and e2
    # e1 in local = (|e1|, 0)
    # e2 in local = (dot(e2, x_axis), dot(e2, y_axis))
    e1_local = np.stack([e1_len[:, 0], np.zeros(len(faces))], axis=1)  # [F, 2]
    e2_local = np.stack([
        np.sum(e2_3d * x_axis, axis=1),
        np.sum(e2_3d * y_axis, axis=1),
    ], axis=1)  # [F, 2]

    # UV edges
    u0 = uv_coords[uv_faces[:, 0]]
    u1 = uv_coords[uv_faces[:, 1]]
    u2 = uv_coords[uv_faces[:, 2]]
    uv_e1 = u1 - u0  # [F, 2]
    uv_e2 = u2 - u0

    # J maps local → UV: J @ [e1_local, e2_local]^T = [uv_e1, uv_e2]^T
    # i.e. J @ M_local = M_uv, so J = M_uv @ inv(M_local)
    det_local = e1_local[:, 0] * e2_local[:, 1] - e1_local[:, 1] * e2_local[:, 0]
    eps = 1e-12
    valid = np.abs(det_local) > eps

    # inv(M_local) for 2x2: [[d, -b], [-c, a]] / det
    a, b = e1_local[:, 0], e1_local[:, 1]
    c, d = e2_local[:, 0], e2_local[:, 1]
    inv_det = np.where(valid, 1.0 / det_local, 0.0)

    M_inv = np.stack([
        np.stack([d * inv_det, -b * inv_det], axis=1),
        np.stack([-c * inv_det, a * inv_det], axis=1),
    ], axis=1)  # [F, 2, 2]

    M_uv = np.stack([uv_e1, uv_e2], axis=1)  # [F, 2, 2]
    J = M_uv @ M_inv  # [F, 2, 2]

    areas_3d = _triangle_areas_3d(vertices, faces)
    return J, areas_3d, valid


def area_distortion_per_face(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray | None,
    uv_faces: np.ndarray | None = None,
) -> np.ndarray:
    """Area distortion per triangle: D = A_3d/A_uv + A_uv/A_3d - 2.

    Both meshes are rescaled so total area = 1 before computing.
    D = 0 means perfect area preservation. Higher = worse.
    Returns [F] float64. Degenerate triangles get NaN.
    """
    uv_c, uv_f = _resolve_uvs(vertices, faces, uv_coords, uv_faces)

    a3d = _triangle_areas_3d(vertices, faces)
    a_uv = np.abs(_triangle_areas_uv(uv_c, uv_f))

    total_3d = a3d.sum() + 1e-12
    total_uv = a_uv.sum() + 1e-12
    a3d_n = a3d / total_3d
    a_uv_n = a_uv / total_uv

    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.where(
            (a3d_n > 0) & (a_uv_n > 0),
            a3d_n / a_uv_n + a_uv_n / a3d_n - 2.0,
            np.nan,
        )
    return d


def angle_distortion_per_face(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray | None,
    uv_faces: np.ndarray | None = None,
) -> np.ndarray:
    """Angle distortion per triangle: D = σ1/σ2 + σ2/σ1 - 2.

    σ1, σ2 are singular values of the Jacobian of the UV map on each triangle.
    D = 0 means conformal (angle-preserving). Higher = worse.
    Returns [F] float64. Degenerate triangles get NaN.
    """
    uv_c, uv_f = _resolve_uvs(vertices, faces, uv_coords, uv_faces)
    J, _, valid = _jacobians(vertices, faces, uv_c, uv_f)

    # singular values via closed-form for 2x2: s = sqrt((tr(J^T J)/2) ± sqrt(...))
    a, b = J[:, 0, 0], J[:, 0, 1]
    c, d = J[:, 1, 0], J[:, 1, 1]
    # eigenvalues of J^T J:  λ = (a²+b²+c²+d²)/2 ± sqrt(((a²+b²+c²+d²)/2)² - (ad-bc)²)
    tr_half = (a**2 + b**2 + c**2 + d**2) / 2.0
    det_sq = (a * d - b * c) ** 2
    disc = np.clip(tr_half**2 - det_sq, 0.0, None)
    sqrt_disc = np.sqrt(disc)

    s1 = np.sqrt(np.clip(tr_half + sqrt_disc, 0.0, None))  # larger SV
    s2 = np.sqrt(np.clip(tr_half - sqrt_disc, 0.0, None))  # smaller SV

    eps = 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        d_angle = np.where(
            valid & (s2 > eps),
            s1 / s2 + s2 / s1 - 2.0,
            np.nan,
        )
    return d_angle


def symmetric_dirichlet_per_face(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray | None,
    uv_faces: np.ndarray | None = None,
) -> np.ndarray:
    """Symmetric Dirichlet energy per face: E = (σ1² + σ2² + 1/σ1² + 1/σ2²) / 2.

    Minimum value = 1 (isometric map). Higher = more distortion.
    Returns [F] float64. Degenerate or flipped triangles get NaN.
    """
    uv_c, uv_f = _resolve_uvs(vertices, faces, uv_coords, uv_faces)
    J, _, valid = _jacobians(vertices, faces, uv_c, uv_f)

    a, b = J[:, 0, 0], J[:, 0, 1]
    c, d = J[:, 1, 0], J[:, 1, 1]
    tr_half = (a**2 + b**2 + c**2 + d**2) / 2.0
    det_sq = np.clip((a * d - b * c) ** 2, 0.0, None)
    disc = np.clip(tr_half**2 - det_sq, 0.0, None)
    sqrt_disc = np.sqrt(disc)

    s1_sq = np.clip(tr_half + sqrt_disc, 0.0, None)
    s2_sq = np.clip(tr_half - sqrt_disc, 0.0, None)

    eps = 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.where(
            valid & (s1_sq > eps) & (s2_sq > eps),
            (s1_sq + s2_sq + 1.0 / s1_sq + 1.0 / s2_sq) / 2.0,
            np.nan,
        )
    return e


def flipped_triangle_percentage(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray | None,
    uv_faces: np.ndarray | None = None,
) -> float:
    """Percentage of triangles that are flipped in UV (det(J) < 0).

    Reports min(x, 100-x) since a global UV mirror swaps the counts.
    Returns a float in [0, 50].
    """
    uv_c, uv_f = _resolve_uvs(vertices, faces, uv_coords, uv_faces)
    signed_areas = _triangle_areas_uv(uv_c, uv_f)
    n_flipped = int(np.sum(signed_areas < 0))
    pct = 100.0 * n_flipped / max(len(faces), 1)
    return min(pct, 100.0 - pct)


def count_uv_shells(
    faces: np.ndarray,
    uv_faces: np.ndarray,
) -> int:


    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_faces = len(faces)
    if n_faces == 0:
        return 0

    # build UV edge -> face list
    uv_edge_to_faces: dict = {}
    for fi, uv_tri in enumerate(uv_faces):
        for k in range(3):
            a, b = int(uv_tri[k]), int(uv_tri[(k + 1) % 3])
            key = (min(a, b), max(a, b))
            uv_edge_to_faces.setdefault(key, []).append(fi)

    rows, cols = [], []
    for face_list in uv_edge_to_faces.values():
        if len(face_list) == 2:
            rows += [face_list[0], face_list[1]]
            cols += [face_list[1], face_list[0]]

    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_faces, n_faces))
    n_components, _ = connected_components(adj, directed=False)
    return int(n_components)


def seam_length(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_faces: np.ndarray,
) -> float:
    """Total 3D length of UV seam edges, normalized by sqrt(total mesh area).

    A seam edge = a mesh edge shared by two faces with different UV indices on the two sides.
    Boundary edges are excluded (they're not seams, they're borders).
    Returns a dimensionless float.
    """
    # build mesh edge -> list of (face_idx, local_edge_pos)
    mesh_edge_to_faces: dict = {}
    for fi, (face, uv_tri) in enumerate(zip(faces, uv_faces)):
        for k in range(3):
            vi, vj = int(face[k]), int(face[(k + 1) % 3])
            ui, uj = int(uv_tri[k]), int(uv_tri[(k + 1) % 3])
            key = (min(vi, vj), max(vi, vj))
            mesh_edge_to_faces.setdefault(key, []).append((fi, ui, uj))

    total_seam_len = 0.0
    for (vi, vj), face_list in mesh_edge_to_faces.items():
        if len(face_list) != 2:
            continue  # boundary edge — skip
        _, ui0, uj0 = face_list[0]
        _, ui1, uj1 = face_list[1]
        # seam if UV edge indices differ on the two sides
        same = (ui0 == ui1 and uj0 == uj1) or (ui0 == uj1 and uj0 == ui1)
        if not same:
            total_seam_len += float(np.linalg.norm(vertices[vj] - vertices[vi]))

    total_area = _triangle_areas_3d(vertices, faces).sum()
    normalization = np.sqrt(total_area + 1e-12)
    return total_seam_len / normalization


def compute_all_uv_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray | None,
    uv_faces: np.ndarray | None = None,
) -> dict:


    if uv_coords is None:
        nan = float('nan')
        return {k: nan for k in [
            'area_distortion_avg', 'area_distortion_max',
            'angle_distortion_avg', 'angle_distortion_max',
            'symmetric_dirichlet_avg', 'flipped_pct',
            'num_shells', 'seam_length',
        ]}

    uv_c, uv_f = _resolve_uvs(vertices, faces, uv_coords, uv_faces)
    areas_3d = _triangle_areas_3d(vertices, faces)
    weight = areas_3d / (areas_3d.sum() + 1e-12)  # area weights for averaging

    def _wavg(arr: np.ndarray) -> float:


        mask = ~np.isnan(arr)
        if not mask.any():
            return float('nan')
        w = weight[mask]
        return float(np.dot(arr[mask], w / (w.sum() + 1e-12)))

    area_d = area_distortion_per_face(vertices, faces, uv_c, uv_f)
    angle_d = angle_distortion_per_face(vertices, faces, uv_c, uv_f)
    dirichlet = symmetric_dirichlet_per_face(vertices, faces, uv_c, uv_f)

    return {
        'area_distortion_avg': _wavg(area_d),
        'area_distortion_max': float(np.nanmax(area_d)) if not np.all(np.isnan(area_d)) else float('nan'),
        'angle_distortion_avg': _wavg(angle_d),
        'angle_distortion_max': float(np.nanmax(angle_d)) if not np.all(np.isnan(angle_d)) else float('nan'),
        'symmetric_dirichlet_avg': _wavg(dirichlet),
        'flipped_pct': flipped_triangle_percentage(vertices, faces, uv_c, uv_f),
        'num_shells': count_uv_shells(faces, uv_f),
        'seam_length': seam_length(vertices, faces, uv_f),
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python evaluation/uv_metrics.py <mesh.obj>')
        sys.exit(1)

    obj_path = sys.argv[1]
    data = parse_obj_with_uv(obj_path)

    verts = data['vertices']
    f = data['faces']
    uv = data['uv_coords']
    uv_f = data['uv_faces']

    print(f'mesh: {Path(obj_path).name}')
    print(f'  vertices: {len(verts)},  faces: {len(f)}')
    if uv is not None:
        print(f'  uv_coords: {len(uv)},  uv_faces: {len(uv_f)}')
    else:
        print('  no UV data found')

    metrics = compute_all_uv_metrics(verts, f, uv, uv_f)

    print('\nUV quality metrics:')
    print(f'  {"metric":<28s}  {"value":>10s}')
    print(f'  {"-"*38}')
    for k, v in metrics.items():
        print(f'  {k:<28s}  {v:>10.4f}')
