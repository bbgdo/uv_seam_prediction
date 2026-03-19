"""
Mesh augmentation via Gaussian vertex perturbation.

Generates N augmented copies of each .obj file by jittering vertex positions.
Face connectivity, UV coordinates, and normals are preserved — only xyz changes.
This lets a small hand-unwrapped dataset grow to a trainable size.

Usage:
    python augment_meshes.py ./3d-objs --copies 3 --noise 0.05
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def _parse_obj_lines(text: str) -> tuple[list[str], list[int]]:
    """Parse OBJ file text, return all lines and indices of 'v ' (vertex position) lines."""
    lines = text.splitlines(keepends=True)
    vertex_indices = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('v ') and not stripped.startswith('vt') and not stripped.startswith('vn'):
            vertex_indices.append(i)
    return lines, vertex_indices


def _parse_vertex_line(line: str) -> np.ndarray:
    """Extract xyz from a 'v x y z' line."""
    parts = line.split()
    return np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)


def _format_vertex_line(coords: np.ndarray) -> str:
    return f'v {coords[0]:.8f} {coords[1]:.8f} {coords[2]:.8f}\n'


def augment_obj_file(
    obj_path: Path,
    n_copies: int,
    noise_fraction: float,
    rng: np.random.Generator,
) -> list[Path]:
    """Create augmented copies of an OBJ file by perturbing vertex positions.

    Directly manipulates OBJ text to guarantee UV preservation.
    Returns list of created file paths.
    """
    text = obj_path.read_text(encoding='utf-8', errors='replace')
    lines, vertex_indices = _parse_obj_lines(text)

    if not vertex_indices:
        print(f"  [skip] {obj_path.name}: no vertex lines found")
        return []

    # extract vertex positions
    vertices = np.array([_parse_vertex_line(lines[i]) for i in vertex_indices])
    bbox_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    noise_scale = noise_fraction * bbox_diag

    created = []
    stem = obj_path.stem
    suffix = obj_path.suffix

    for copy_idx in range(n_copies):
        noise = rng.normal(0, noise_scale, vertices.shape)
        perturbed = vertices + noise

        new_lines = lines.copy()
        for line_idx, v_idx in zip(vertex_indices, range(len(perturbed))):
            new_lines[line_idx] = _format_vertex_line(perturbed[v_idx])

        out_path = obj_path.parent / f'{stem}_aug{copy_idx}{suffix}'
        out_path.write_text(''.join(new_lines), encoding='utf-8')
        created.append(out_path)

    return created


def main():
    parser = argparse.ArgumentParser(description='Augment meshes via Gaussian vertex perturbation.')
    parser.add_argument('mesh_dir', help='Directory containing .obj files')
    parser.add_argument('--copies', type=int, default=3, help='Augmented copies per mesh (default: 3)')
    parser.add_argument('--noise', type=float, default=0.05, help='Noise as fraction of bbox diagonal (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir)
    if not mesh_dir.is_dir():
        print(f"[error] directory not found: {mesh_dir}")
        sys.exit(1)

    # only process original files, skip already-augmented ones
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
