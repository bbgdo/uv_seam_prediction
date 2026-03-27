# UV Seam Predictor — ML Pipeline

An end-to-end pipeline for automatically placing UV seams on 3D meshes using Graph Neural Networks, deployed as a Blender add-on. Three architectures are compared: DualGraphSAGE, DualGATv2 (both on the dual graph), and MeshCNN (fixed 4-neighbor edge convolution on the original mesh).

## Overview

UV seam placement is a tedious manual task in 3D modeling. This project frames it as a **binary edge classification problem**: given a triangulated mesh, predict which edges should be UV seam cuts.

## Project Structure

```
dataset_py_utils/
├── preprocessing/           # Mesh cleanup, feature engineering, augmentation,
│                               graph conversion, and dual graph construction
├── models/                  # GNN architectures, training, and experiment logging
│   ├── dual_graphsage/      # DualGraphSAGE node classifier (dual graph)
│   ├── gatv2/               # DualGATv2 node classifier (dual graph)
│   ├── meshcnn/             # MeshCNN edge classifier (original mesh, fixed 4-neighbor conv)
│   └── utils/               # Dataset loading, metrics, losses, post-processing, logging
├── evaluation/              # UV-level evaluation: unwrap quality metrics + comparison plots
├── runs/                    # Experiment outputs (JSON logs, plots, checkpoints)
└── blender_bridge/          # Blender add-on for running inference
```

## Pipeline

```
Raw 3D files
    → [preprocessing]                    cleanup, format conversion, scale normalization
    → [augment_meshes.py]                data augmentation (Gaussian vertex perturbation)
    → [obj_to_dataset_graph.py]              build original graph dataset (dataset.pt)
    → [build_dual_graph.py]                  build dual graph dataset (dataset_dual.pt)
    → [build_meshcnn_data.py]                build MeshCNN dataset (dataset_meshcnn.pt)
    → [models/dual_graphsage/train.py]       train DualGraphSAGE on dual graph
    → [models/gatv2/train.py]                train DualGATv2 on dual graph
    → [models/meshcnn/train.py]              train MeshCNN on original mesh
    → [evaluation/run_evaluation.py]         UV-level evaluation (unwrap + metrics)
    → [evaluation/compare_models.py]         cross-model comparison tables + plots
    → [blender_bridge]                       load weights, run inference inside Blender
```

## Graph Representation

### Dataset format (`dataset.pt`)

Each mesh is stored as a PyTorch Geometric `Data` object (original graph, used for feature engineering):

| Tensor | Shape | Description |
|---|---|---|
| `x` | `[N, 6]` | vertex coords + normals |
| `edge_index` | `[2, 2E]` | all edges stored both directions |
| `edge_attr` | `[2E, 11]` | 11-dim feature vector (see below) |
| `y` | `[2E]` | 1 = seam, 0 = not a seam |
| `faces` | `[F, 3]` | triangle face indices |

### Dual graph (`dataset_dual.pt`)

Each original edge becomes a dual node; two dual nodes are connected if their original edges share a face.

| Tensor | Shape | Description |
|---|---|---|
| `x` | `[E, 11]` | original edge features become dual node features |
| `edge_index` | `[2, D]` | face-adjacency connectivity |
| `y` | `[E]` | seam labels per dual node |

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

### MeshCNN dataset (`dataset_meshcnn.pt`)

MeshCNN operates on original mesh topology using a fixed 4-neighbor structure. Run `preprocessing/build_meshcnn_data.py` to convert `dataset.pt`:

```bash
python preprocessing/build_meshcnn_data.py --input dataset.pt --output dataset_meshcnn.pt
```

| Tensor | Shape | Description |
|---|---|---|
| `x` | `[E, 11]` | 11-dim edge features (same as `dataset.pt`) |
| `edge_neighbors` | `[E, 4]` | 4 neighboring edge indices per edge (-1 = boundary) |
| `y` | `[E]` | seam labels |

Interior edges have all 4 neighbors; boundary edges have -1 in positions 2–3.

---

## Experiment Logging

Training runs produce structured outputs in `runs/<experiment_name>/`:

| File | Description |
|---|---|
| `config.json` | Hyperparameters and dataset stats |
| `metrics.json` | Per-epoch train/val loss, F1, precision, recall, LR |
| `summary.json` | Best epoch, test metrics, timing |
| `loss_curves.png` | Train/val loss over epochs |
| `f1_curves.png` | Train/val F1 with best epoch marker |
| `precision_recall_curves.png` | Val precision vs recall over epochs |
| `lr_schedule.png` | Learning rate schedule (log scale) |
| `class_balance.png` | Seam vs non-seam counts per split |
| `best_model.pth` | Best model checkpoint (by val F1) |

Compare experiments:
```bash
python models/utils/comparison.py runs/dual_graphsage_001 runs/gatv2_001 runs/meshcnn_001
```
Generates `comparison_f1.png` (overlaid F1 curves) and `comparison_table.png` (test results table).

## Training Options

### Connectivity loss (`--lambda-conn`)

All three training scripts accept `--lambda-conn` (default `0.0`). When set, a differentiable penalty is added to the BCE loss that discourages isolated seam predictions on the dual graph — dual nodes with high seam probability but low-probability neighbors are penalized.

```bash
python models/dual_graphsage/train.py --dataset dataset_dual.pt --lambda-conn 0.1
```

### Post-processing

`models/utils/postprocess.py` applies inference-time cleanup to raw model probabilities:
1. **Threshold + clean** — removes disconnected seam components smaller than `--min-component` edges.
2. **Stitch gaps** — greedily bridges small gaps between disconnected seam components.

```bash
python models/utils/postprocess.py \
    --dataset dataset.pt --dual-dataset dataset_dual.pt \
    --weights runs/dual_graphsage_001/best_model.pth \
    --threshold 0.5 --min-component 3 --max-gap 3
```

## Requirements

- Python 3.10+
- `torch`, `torch-geometric`, `trimesh`, `scipy`, `matplotlib`
- Blender 4.5 LTS (might work with Blender 4.0+) (for preprocessing scripts and the add-on)
- Optional: `pyembree` (faster AO raycasting; falls back to normal-based approximation without it)
