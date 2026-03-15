# Preprocessing

Blender scripts that turn raw downloaded mesh files into a clean, normalized `.obj` dataset ready for graph conversion. Run these in order.

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
