# Blender Bridge

> **TEMPORARY: To be replaced with a more robust solution in the final version**

A Blender add-on that runs trained GNN inference and marks predicted UV seams on the active mesh.

---

## Why a subprocess?

PyTorch's `c10.dll` conflicts with Blender's OpenMP DLLs on Windows, making it impossible to `import torch` inside Blender's Python process. The add-on solves this by spawning an **external Python subprocess** that loads torch, runs inference, and writes results back via temp files (`.npz` in, `.txt` out). The Blender process itself only uses `numpy`.

---

## Setup

1. Install the add-on in Blender: **Edit → Preferences → Add-ons → Install from disk**, select `blender_bridge/__init__.py` (or zip the folder first).
2. In the **N-panel → UV Seam GNN**, set **Python Exe** to the Python executable that has `torch`, `torch_geometric`, and `scipy` installed (e.g. `C:/Users/you/.venv/Scripts/python.exe`).
3. Click **Test Python** to verify the imports work.
4. Set **Weights** to your `best_model.pth` checkpoint.
5. Set **Model** to match the architecture the weights were trained with (`DualGraphSAGE` or `DualGATv2`).
6. Select a mesh object and click **Auto-Mark UV Seams**.

See `INSTALL.md` for troubleshooting common setup issues.

---

## Components

### `__init__.py` — Add-on Entry Point

<details>
<summary>Click to expand: class breakdown</summary>

| Class | Role |
|---|---|
| `UVSeamGNNProperties` | N-panel settings: Python exe path, weights path, model type, sigmoid threshold |
| `OBJECT_OT_test_uv_seam_python` | "Test Python" button — runs a quick `import torch, torch_geometric` check |
| `OBJECT_OT_predict_uv_seams` | Main operator — extracts raw mesh geometry, spawns subprocess, marks seam edges |
| `VIEW3D_PT_uv_seam_gnn` | N-panel UI layout |

`_mesh_to_arrays()` extracts raw geometry from the active Blender mesh using BMesh (triangulates on-the-fly, no OBJ export needed). Returns:

| Array | Shape | Description |
|---|---|---|
| `vertices` | `[N, 3]` | vertex positions |
| `normals` | `[N, 3]` | vertex normals |
| `faces` | `[F, 3]` | triangulated face vertex indices |
| `unique_edges` | `[E, 2]` | undirected edges (vi < vj) |

Feature computation is delegated entirely to the subprocess, keeping the Blender side dependency-free.

</details>

### `run_inference.py` — Inference Worker

Standalone script that runs as the external subprocess. **Fully self-contained** — no imports from the project tree. Embeds:

- 11-dim edge feature computation (pure numpy, from raw geometry arrays)
- Dual graph construction (face-adjacency edge_index)
- `DualGraphSAGE` model definition (SAGEConv + LayerNorm + residuals)
- `DualGATv2` model definition (GATv2Conv + multi-head attention + LayerNorm + residuals)
- Post-processing: small-component removal + greedy gap stitching

```
python run_inference.py <data.npz> <weights.pth> <threshold> <output.txt>
                        [--model-type graphsage|gatv2]
                        [--min-component N]
                        [--max-gap N]
```

Reads raw mesh geometry from `.npz`, computes features, builds the dual graph, runs the model, applies post-processing, and writes predicted seam edge indices (one per line, 0-based into the unique-edge list) to the output file.

---

## Data Flow

```
Blender mesh
    → _mesh_to_arrays()        extract raw geometry (numpy only, no torch)
    → mesh.npz                 vertices, normals, faces, unique_edges
    → subprocess: run_inference.py
        → compute 11 edge features (pure numpy)
        → build dual graph (face adjacency)
        → load DualGraphSAGE or DualGATv2 weights
        → run GNN inference
        → threshold + clean small components + stitch gaps
    → seams.txt                predicted seam edge indices
    → mark edge.use_seam = True on each predicted edge
```
