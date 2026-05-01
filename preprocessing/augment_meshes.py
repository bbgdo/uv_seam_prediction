import argparse
import re
import sys
from pathlib import Path

import numpy as np


MIN_EDGE_LENGTH = 1e-12


def _parse_obj_lines(text: str) -> tuple[list[str], list[int], list[int]]:
    lines = text.splitlines(keepends=True)
    vertex_indices = []
    face_indices = []

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('v ') and not stripped.startswith('vt') and not stripped.startswith('vn'):
            vertex_indices.append(i)
        elif stripped.startswith('f '):
            face_indices.append(i)

    return lines, vertex_indices, face_indices


def _parse_vertex_line(line: str) -> np.ndarray:
    parts = line.split()
    return np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)


def _format_vertex_line(coords: np.ndarray) -> str:
    return f'v {coords[0]:.8f} {coords[1]:.8f} {coords[2]:.8f}\n'


def _parse_obj_vertex_index(token: str, n_vertices: int) -> int | None:
    raw_index = token.split('/', 1)[0]
    if not raw_index:
        return None

    try:
        obj_index = int(raw_index)
    except ValueError:
        return None

    index = obj_index - 1 if obj_index > 0 else n_vertices + obj_index
    return index if 0 <= index < n_vertices else None


def _compute_local_vertex_scale(lines: list[str], face_indices: list[int], vertices: np.ndarray) -> np.ndarray:
    n_vertices = len(vertices)
    bbox_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    fallback = max(float(bbox_diag), MIN_EDGE_LENGTH)
    edges = set()

    for line_idx in face_indices:
        face = []
        for token in lines[line_idx].split()[1:]:
            index = _parse_obj_vertex_index(token, n_vertices)
            if index is None:
                face = []
                break
            face.append(index)

        if len(face) < 2:
            continue

        for a, b in zip(face, face[1:] + face[:1]):
            if a != b:
                edges.add(tuple(sorted((a, b))))

    if not edges:
        return np.full(n_vertices, fallback, dtype=np.float64)

    edges = np.asarray(list(edges), dtype=np.int64)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    valid = edge_lengths > MIN_EDGE_LENGTH

    if not np.any(valid):
        return np.full(n_vertices, fallback, dtype=np.float64)

    edges = edges[valid]
    edge_lengths = edge_lengths[valid]

    scale_sum = np.zeros(n_vertices, dtype=np.float64)
    degree = np.zeros(n_vertices, dtype=np.int64)
    u = edges[:, 0]
    v = edges[:, 1]

    np.add.at(scale_sum, u, edge_lengths)
    np.add.at(scale_sum, v, edge_lengths)
    np.add.at(degree, u, 1)
    np.add.at(degree, v, 1)

    local_scale = np.divide(
        scale_sum,
        degree,
        out=np.full(n_vertices, np.nan, dtype=np.float64),
        where=degree > 0,
    )
    local_scale[~np.isfinite(local_scale)] = float(np.median(edge_lengths))
    return np.maximum(local_scale, MIN_EDGE_LENGTH)


def augment_obj_file(
    obj_path: Path,
    n_copies: int,
    noise_fraction: float,
    rng: np.random.Generator,
) -> list[Path]:
    """Directly manipulates OBJ text to guarantee UV preservation."""
    text = obj_path.read_text(encoding='utf-8', errors='replace')
    lines, vertex_indices, face_indices = _parse_obj_lines(text)

    if not vertex_indices:
        print(f"  [skip] {obj_path.name}: no vertex lines found")
        return []

    vertices = np.array([_parse_vertex_line(lines[i]) for i in vertex_indices])
    local_scale = _compute_local_vertex_scale(lines, face_indices, vertices)

    created = []
    stem = obj_path.stem
    suffix = obj_path.suffix

    for copy_idx in range(n_copies):
        noise = rng.normal(0.0, noise_fraction, vertices.shape) * local_scale[:, None]
        perturbed = vertices + noise

        new_lines = lines.copy()
        for line_idx, v_idx in zip(vertex_indices, range(len(perturbed))):
            new_lines[line_idx] = _format_vertex_line(perturbed[v_idx])

        out_path = obj_path.parent / f'{stem}_aug{copy_idx}{suffix}'
        out_path.write_text(''.join(new_lines), encoding='utf-8')
        created.append(out_path)

    return created


def main():
    parser = argparse.ArgumentParser(description='Augment meshes via density-aware Gaussian vertex perturbation.')
    parser.add_argument('mesh_dir', help='Directory containing .obj files')
    parser.add_argument('--copies', type=int, default=3, help='Augmented copies per mesh (default: 3)')
    parser.add_argument('--noise', type=float, default=0.05, help='Gaussian noise as fraction of local mean edge length (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir)
    if not mesh_dir.is_dir():
        print(f"[error] directory not found: {mesh_dir}")
        sys.exit(1)

    obj_files = sorted([
        f for f in mesh_dir.glob('*.obj')
        if not re.search(r'_aug\d+\.obj$', f.name)
    ])

    if not obj_files:
        print(f"[error] no .obj files found in {mesh_dir}")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    total_created = 0

    print(f"augmenting {len(obj_files)} mesh(es) with {args.copies} copies each (noise={args.noise})...\n")

    for obj_path in obj_files:
        created = augment_obj_file(obj_path, args.copies, args.noise, rng)
        total_created += len(created)
        print(f"  {obj_path.name} -> {len(created)} augmented copies")

    total_meshes = len(obj_files) + total_created
    print(f"\ndone. created {total_created} augmented files.")
    print(f"total meshes in {mesh_dir}: {total_meshes} ({len(obj_files)} original + {total_created} augmented)")


if __name__ == '__main__':
    main()
