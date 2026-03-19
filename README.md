# UV Seam Predictor — ML Pipeline

> ! outdated info and code. to be updated

An end-to-end pipeline for automatically placing UV seams on 3D meshes using a Graph Neural Network (GraphSAGE), deployed as a Blender add-on.

## Overview

UV seam placement is a tedious manual task in 3D modeling. This project frames it as a **binary edge classification problem**: given a triangulated mesh, predict which edges should be UV seam cuts.

## Project Structure

```
dataset_py_utils/
├── preprocessing/           # Mesh cleanup, feature engineering, augmentation,
│                               and OBJ → PyG graph conversion
├── models/                  # GNN architecture and training logic
│   ├── graphsage/           # GraphSAGE model + training script
│   └── utils/               # Dataset loading, splitting, and metrics
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
| `edge_attr` | `[2E, 11]` | 11-dim feature vector (see below) |
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

<details>
<summary>Click to expand: Edge features (11-dim)</summary>

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 0 | `edge_length` | [0, 1] | Euclidean distance, normalized by max edge length per mesh |
| 1 | `signed_dihedral` | [-1, 1] | Dihedral angle / pi. Positive = convex, negative = concave |
| 2 | `sharpness` | [0, 1] | abs(signed_dihedral). 0 = flat, 1 = knife-edge |
| 3 | `concavity` | [-1, 1] | Same as signed_dihedral (signed sharpness) |
| 4 | `delta_normal` | [0, 1] | Vertex normal difference magnitude / 2 |
| 5 | `dot_normal` | [-1, 1] | Dot product of endpoint vertex normals |
| 6 | `gauss_curv_mean` | [-1, 1] | Mean Gaussian curvature of endpoints (z-score normalized) |
| 7 | `gauss_curv_diff` | [0, 2] | Absolute difference in Gaussian curvature between endpoints |
| 8 | `ao_mean` | [0, 1] | Mean ambient occlusion of endpoints |
| 9 | `ao_diff` | [0, 1] | Absolute AO difference between endpoints |
| 10 | `symmetry_dist` | [0, 1] | Edge midpoint distance to detected symmetry plane |

Feature computation is implemented in `preprocessing/compute_features.py`.

</details>

<details>
<summary>Click to expand: Data augmentation</summary>

`preprocessing/augment_meshes.py` creates augmented copies of meshes by adding Gaussian noise to vertex positions while preserving topology, face connectivity, and UV coordinates. This multiplies the dataset size without requiring additional manual UV unwraps.

```bash
python preprocessing/augment_meshes.py ./3d-objs --copies 3 --noise 0.05
```

</details>

## Requirements

- Python 3.10+
- `torch`, `torch-geometric`, `trimesh`, `scipy`
- Blender 4.5 LTS (might work with Blender 4.0+) (for preprocessing scripts and the add-on)
- Optional: `pyembree` (faster AO raycasting; falls back to normal-based approximation without it)
