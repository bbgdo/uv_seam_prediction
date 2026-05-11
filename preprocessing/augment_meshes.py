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


def _compute_shift_offsets(
    vertices: np.ndarray,
    n_zones: int,
    radius: float,
    falloff: float,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_zones <= 0:
        return np.zeros_like(vertices)
    if radius <= 0.0:
        raise ValueError('shift radius must be positive')
    if falloff <= 0.0:
        raise ValueError('shift falloff must be positive')
    if strength < 0.0:
        raise ValueError('shift strength must be non-negative')

    offsets = np.zeros_like(vertices)
    center_indices = rng.integers(0, len(vertices), size=n_zones)
    centers = vertices[center_indices]
    zone_shifts = rng.normal(0.0, strength, size=(n_zones, 3))

    for center, zone_shift in zip(centers, zone_shifts):
        distances = np.linalg.norm(vertices - center, axis=1)
        weights = np.clip(1.0 - distances / radius, 0.0, 1.0)
        weights = weights * weights * (3.0 - 2.0 * weights)
        weights = weights ** falloff
        offsets += weights[:, None] * zone_shift[None, :]

    return offsets


def augment_obj_file(
    obj_path: Path,
    n_copies: int,
    enable_noise: bool,
    noise_fraction: float | None,
    enable_shift: bool,
    shift_zones: int,
    shift_radius: float,
    shift_falloff: float,
    shift_strength: float,
    rng: np.random.Generator,
) -> list[Path]:
    text = obj_path.read_text(encoding='utf-8', errors='replace')
    lines, vertex_indices, face_indices = _parse_obj_lines(text)

    if not vertex_indices:
        print(f"  [skip] {obj_path.name}: no vertex lines found")
        return []

    vertices = np.array([_parse_vertex_line(lines[i]) for i in vertex_indices])
    local_scale = _compute_local_vertex_scale(lines, face_indices, vertices) if enable_noise else None

    created = []
    stem = obj_path.stem
    suffix = obj_path.suffix

    for copy_idx in range(n_copies):
        offsets = np.zeros_like(vertices)

        if enable_noise:
            offsets += rng.normal(0.0, noise_fraction, vertices.shape) * local_scale[:, None]

        if enable_shift:
            offsets += _compute_shift_offsets(
                vertices=vertices,
                n_zones=shift_zones,
                radius=shift_radius,
                falloff=shift_falloff,
                strength=shift_strength,
                rng=rng,
            )

        perturbed = vertices + offsets

        new_lines = lines.copy()
        for line_idx, v_idx in zip(vertex_indices, range(len(perturbed))):
            new_lines[line_idx] = _format_vertex_line(perturbed[v_idx])

        out_path = obj_path.parent / f'{stem}_aug{copy_idx}{suffix}'
        out_path.write_text(''.join(new_lines), encoding='utf-8')
        created.append(out_path)

    return created


def main():
    parser = argparse.ArgumentParser(description='Augment meshes via relative vertex noise and/or smooth regional shifts.')
    parser.add_argument('mesh_dir', help='Directory containing .obj files')
    parser.add_argument('--copies', type=int, default=3, help='Augmented copies per mesh (default: 3)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    parser.add_argument('--enable-noise', action='store_true', help='Enable density-aware Gaussian vertex noise')
    parser.add_argument('--noise', type=float, default=None, help='Gaussian noise as fraction of local mean edge length')

    parser.add_argument('--enable-shift', action='store_true', help='Enable smooth regional vertex shifts')
    parser.add_argument('--shift-zones', type=int, default=4, help='Number of random shift zones per copy (default: 4)')
    parser.add_argument('--shift-radius', type=float, default=0.15, help='Shift zone radius in mesh units (default: 0.15)')
    parser.add_argument('--shift-falloff', type=float, default=2.0, help='Shift falloff exponent; larger values localize the shift more strongly (default: 2.0)')
    parser.add_argument('--shift-strength', type=float, default=0.02, help='Gaussian shift vector std in mesh units (default: 0.02)')
    args = parser.parse_args()

    if not args.enable_noise and not args.enable_shift:
        print('[error] enable at least one augmentation mode: --enable-noise and/or --enable-shift')
        sys.exit(1)
    if args.enable_noise and args.noise is None:
        print('[error] --enable-noise requires --noise')
        sys.exit(1)
    if not args.enable_noise and args.noise is not None:
        print('[error] --noise was provided but --enable-noise is not enabled')
        sys.exit(1)
    if args.enable_noise and args.noise < 0.0:
        print('[error] --noise must be non-negative')
        sys.exit(1)
    if args.enable_shift and args.shift_zones <= 0:
        print('[error] --shift-zones must be positive')
        sys.exit(1)
    if args.enable_shift and args.shift_radius <= 0.0:
        print('[error] --shift-radius must be positive')
        sys.exit(1)
    if args.enable_shift and args.shift_falloff <= 0.0:
        print('[error] --shift-falloff must be positive')
        sys.exit(1)
    if args.enable_shift and args.shift_strength < 0.0:
        print('[error] --shift-strength must be non-negative')
        sys.exit(1)

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
    modes = []
    if args.enable_noise:
        modes.append(f'noise={args.noise}')
    if args.enable_shift:
        modes.append(
            f'shift_zones={args.shift_zones}, shift_radius={args.shift_radius}, '
            f'shift_falloff={args.shift_falloff}, shift_strength={args.shift_strength}'
        )

    print(f"augmenting {len(obj_files)} mesh(es) with {args.copies} copies each ({'; '.join(modes)})...\n")

    for obj_path in obj_files:
        created = augment_obj_file(
            obj_path=obj_path,
            n_copies=args.copies,
            enable_noise=args.enable_noise,
            noise_fraction=args.noise,
            enable_shift=args.enable_shift,
            shift_zones=args.shift_zones,
            shift_radius=args.shift_radius,
            shift_falloff=args.shift_falloff,
            shift_strength=args.shift_strength,
            rng=rng,
        )
        total_created += len(created)
        print(f"  {obj_path.name} -> {len(created)} augmented copies")

    total_meshes = len(obj_files) + total_created
    print(f"\ndone. created {total_created} augmented files.")
    print(f"total meshes in {mesh_dir}: {total_meshes} ({len(obj_files)} original + {total_created} augmented)")


if __name__ == '__main__':
    main()
