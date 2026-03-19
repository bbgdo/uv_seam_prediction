# Blender Bridge

> **TEMPORARY: To be replaced with a more robust solution in the final version**

A Blender add-on that runs trained GNN inference and marks predicted UV seams on the active mesh.

---

## Why a subprocess?

PyTorch's `c10.dll` conflicts with Blender's OpenMP DLLs on Windows, making it impossible to `import torch` inside Blender's Python process. The add-on solves this by spawning an **external Python subprocess** that loads torch, runs inference, and writes results back via temp files (`.npz` in, `.txt` out). The Blender process itself only uses `numpy`.

---

## Setup

1. Install the add-on in Blender: **Edit → Preferences → Add-ons → Install from disk**, select `blender_bridge/__init__.py` (or zip the folder first).
2. In the **N-panel → UV Seam GNN**, set **Python Exe** to the Python executable that has `torch` and `torch_geometric` installed (e.g. `C:/Users/you/.venv/Scripts/python.exe`).
3. Click **Test Python** to verify the imports work.
4. Set **Weights** to your `best_model.pth` checkpoint.
5. Select a mesh object and click **Auto-Mark UV Seams**.

See `INSTALL.md` for troubleshooting common setup issues.

---

## Components

### `__init__.py` — Add-on Entry Point

<details>
<summary>Click to expand: class breakdown</summary>

| Class | Role |
|---|---|
| `UVSeamGNNProperties` | N-panel settings: Python exe path, weights path, sigmoid threshold |
| `OBJECT_OT_test_uv_seam_python` | "Test Python" button — runs a quick `import torch, torch_geometric` check |
| `OBJECT_OT_predict_uv_seams` | Main operator — builds graph, spawns subprocess, marks seam edges |
| `VIEW3D_PT_uv_seam_gnn` | N-panel UI layout |

`_mesh_to_arrays()` extracts vertex positions, normals, and edge features directly from the active Blender mesh using BMesh. It triangulates on the fly, so no OBJ export is needed.

</details>

### `run_inference.py` — Inference Worker

Standalone script that runs as the external subprocess. It embeds its own copy of `UVSeamGNN` (3 SAGEConv layers with residual, `edge_in_dim=11`) so it works from any installation path without requiring the project root on `PYTHONPATH`.

```
python run_inference.py <data.npz> <weights.pth> <threshold> <output.txt>
```

Reads the mesh graph from `.npz`, runs the model, and writes predicted seam edge indices (one per line, 0-based into the unique-edge list) to the output file.

---

## Data Flow

```
Blender mesh
    → _mesh_to_arrays()      extract graph (numpy only, no torch)
    → mesh.npz               written to a temp directory
    → subprocess: run_inference.py
    → seams.txt              predicted edge indices
    → mark edge.use_seam = True on each predicted edge
```
