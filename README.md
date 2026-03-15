# UV Seam Predictor — ML Pipeline

An end-to-end pipeline for automatically placing UV seams on 3D meshes using a Graph Neural Network (GraphSAGE), deployed as a Blender add-on.

## Overview

UV seam placement is a tedious manual task in 3D modeling. This project frames it as a **binary edge classification problem**: given a triangulated mesh, predict which edges should be UV seam cuts.

## Project Structure

```
dataset_py_utils/
├── preprocessing/           # Blender scripts that clean and prepare raw mesh data 
│                               and converts OBJ meshes into a PyG graph dataset
├── models/                  # GNN architecture and training logic
│   ├── graphsage/           # GraphSAGE model + training script (placeholder)
│   └── utils/               # Dataset loading, splitting, and metrics (placeholder)
└── blender_bridge/          # Blender add-on for running inference
```

## Pipeline

```
Raw 3D files
    → [preprocessing]           cleanup, format conversion, scale normalization
    → [obj_to_dataset_graph.py] build graph dataset (dataset.pt)
    → [models/graphsage/train.py] train GNN
    → [blender_bridge]          load weights, run inference inside Blender
```

## Graph Representation

Each mesh becomes a directed graph stored as a PyTorch Geometric `Data` object:

| Tensor | Shape | Description |
|---|---|---|
| `x` | `[N, 6]` | vertex coords + normals |
| `edge_index` | `[2, 2E]` | all edges stored both directions |
| `edge_attr` | `[2E, 4]` | length, dihedral angle, Δnormal, dot(normals) |
| `y` | `[2E]` | 1 = seam, 0 = not a seam |

<details>
<summary>Click to expand: Building the dataset</summary>

```bash
python obj_to_dataset_graph.py ./meshes --max-meshes 200 --save
```

Scans `./meshes` for `.obj` files, converts each to a PyG `Data` object, prints per-mesh statistics and class balance, then saves the full list to `dataset.pt`.

Meshes with zero detected seam edges are flagged as outliers and excluded.

Seam detection works on actual UV data when present — an edge is a seam if either endpoint has different UV coordinates across its two adjacent faces. Boundary edges are always seams. Falls back to boundary-only detection when the mesh has no UVs.

</details>

## Requirements

- Python 3.10+
- `torch`, `torch-geometric`, `trimesh`
- Blender 4.5 LTS (might work with Blender 4.0+) (for preprocessing scripts and the add-on)
