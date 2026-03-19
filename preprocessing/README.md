# Preprocessing

Blender scripts that turn raw downloaded mesh files into a clean, normalized `.obj` dataset ready for graph conversion, plus feature engineering and data augmentation tools.

---

## Scripts

### 1. `select_valid_models.py` — Dataset Filtering

Utility script (plain Python, not a Blender script) that reads a whitelist of valid filenames from `valid_files.txt` and deletes everything else from the target directory. Run this after manual review to cull bad or unsuitable meshes.

Configure `TARGET_DIR` and `WHITELIST` at the bottom of the file, then:

```bash
python select_valid_models.py
```

---

### 2. `convert_to_obj.py` — Format Conversion

Converts FBX, DAE, GLB, GLTF, BLEND, and OBJ files to triangulated OBJ format. Files that are already `.obj` are copied to an `already_obj/` subfolder unchanged.

Output directory: `{input_dir}_OBJ/`

<details>
<summary>Click to expand: Usage</summary>

```bash
blender -b --factory-startup -P convert_to_obj.py -- "path/to/Mesh_Files" > convert_to_obj_logs.txt
```

</details>

---

### 3. `cleanup_machin3.py` — Geometry Cleanup

Loads each `.obj` and runs a sequence of mesh repair operations via Blender's edit-mode operators:

- Remove duplicate vertices (threshold = 0.0001)
- Delete loose geometry
- Dissolve degenerate faces
- Recalculate normals (outward)
- Triangulate quads and n-gons

Output directory: `{input_dir}_machine_cleaned/`

<details>
<summary>Click to expand: Usage</summary>

```bash
blender -b --factory-startup -P cleanup_machin3.py -- "path/to/Mesh_Files_OBJ" > cleanup_logs.txt
```

</details>

---

### 4. `normalize_scale.py` — Scale Normalization

Centers each mesh at the origin and scales it so its bounding box diagonal equals 1.0. Final vertex coordinates land in roughly `[-0.5, 0.5]` on each axis. Transforms are baked into the mesh data before export so the exported `.obj` has raw normalized coordinates.

Output directory: `{input_dir}_normalized/`

<details>
<summary>Click to expand: Usage</summary>

```bash
blender -b --factory-startup -P normalize_scale.py -- "path/to/Mesh_Files_machine_cleaned" > normalization_logs.txt
```

</details>

---

### 5. `augment_meshes.py` — Data Augmentation

Creates augmented copies of `.obj` files by adding Gaussian noise to vertex positions. Face connectivity, UV coordinates, and normals are preserved — only vertex XYZ changes. Uses text-level OBJ manipulation to guarantee UV preservation.

```bash
python augment_meshes.py ./3d-objs --copies 3 --noise 0.05
```

| Flag | Default | Description |
|---|---|---|
| `mesh_dir` | (required) | Directory containing `.obj` files |
| `--copies` | 3 | Number of augmented copies per mesh |
| `--noise` | 0.05 | Noise magnitude as fraction of bounding box diagonal |
| `--seed` | 42 | Random seed |

Already-augmented files (matching `*_aug*.obj`) are skipped on re-runs.

---

### 6. `compute_features.py` — Edge Feature Engineering

Computes 11 edge-level features from a `trimesh.Trimesh` object. Each feature function is standalone and testable. Used by `obj_to_dataset_graph.py` but can also run standalone for inspection:

```bash
python compute_features.py path/to/mesh.obj
```

Prints per-feature statistics (min, max, mean, std) and checks for NaN/Inf.

<details>
<summary>Click to expand: Feature list</summary>

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 0 | `edge_length` | [0, 1] | Normalized by max edge length per mesh |
| 1 | `signed_dihedral` | [-1, 1] | Dihedral angle / pi (positive = convex) |
| 2 | `sharpness` | [0, 1] | abs(signed_dihedral) |
| 3 | `concavity` | [-1, 1] | Signed sharpness |
| 4 | `delta_normal` | [0, 1] | Vertex normal difference magnitude / 2 |
| 5 | `dot_normal` | [-1, 1] | Dot product of endpoint vertex normals |
| 6 | `gauss_curv_mean` | [-1, 1] | Mean Gaussian curvature (angle defect, z-score normalized) |
| 7 | `gauss_curv_diff` | [0, 2] | Absolute curvature difference at endpoints |
| 8 | `ao_mean` | [0, 1] | Mean ambient occlusion at endpoints |
| 9 | `ao_diff` | [0, 1] | AO difference at endpoints |
| 10 | `symmetry_dist` | [0, 1] | Edge midpoint distance to symmetry plane |

AO uses raycasting (pyembree > ray_triangle) when available, otherwise falls back to a normal-based approximation. Symmetry detection uses `scipy.spatial.cKDTree` for mirror-vertex matching.

</details>

---

### 7. `obj_to_dataset_graph.py` — Graph Dataset Builder

Converts `.obj` meshes into PyG `Data` objects with the full 11-dim edge feature vector and face indices. Seam labels are derived from UV coordinate splits across adjacent faces.

```bash
python obj_to_dataset_graph.py ./3d-objs --max-meshes 50 --save
```

| Flag | Default | Description |
|---|---|---|
| `mesh_dir` | `./meshes` | Directory with `.obj` files |
| `--max-meshes` | 5 | Max meshes to process |
| `--save` | off | Save dataset as `dataset.pt` |

---

### 8. `build_dual_graph.py` — Dual Graph Construction

Converts the original graph dataset into a dual (line) graph for GATv2 training. Each original edge becomes a dual node, with dual edges connecting edges that share a face.

```bash
python build_dual_graph.py --input dataset.pt --output dataset_dual.pt
```

| Flag | Description |
|---|---|
| `--input` | Path to original `dataset.pt` |
| `--output` | Path to save dual dataset |
