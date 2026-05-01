# UV Seam Predictor — ML Pipeline

An end-to-end pipeline for automatically placing UV seams on 3D meshes using Graph Neural Networks, deployed as a Blender add-on. Three architectures are compared: DualGraphSAGE, DualGATv2 (both on the dual graph), and SparseMeshCNN (fixed 4-neighbor edge convolution on the original mesh).

## Overview

UV seam placement is a tedious manual task in 3D modeling. This project frames it as a **binary edge classification problem**: given a triangulated mesh, predict which edges should be UV seam cuts.

## Project Structure

```
dataset_py_utils/
├── preprocessing/           # Mesh cleanup, feature engineering, augmentation,
│                               graph conversion and GNN dataset preparation
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
    → [obj_to_dataset_graph.py]              build the maintained PyG GNN dataset entrypoint
    → [build_meshcnn_dataset_v2.py]          build SparseMeshCNN `MeshCNNSample` dataset
    → [models/dual_graphsage/train.py]       train DualGraphSAGE on dual graph
    → [models/gatv2/train.py]                train DualGATv2 on dual graph
    → [models/meshcnn_full/train.py]         train SparseMeshCNN on original mesh
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

### Dual graph view

`preprocessing/obj_to_dataset_graph.py` is the maintained GNN/PyG builder. It writes the canonical edge-level dataset and exports the dual-view helpers used by GraphSAGE/GATv2 tooling.

| Tensor | Shape | Description |
|---|---|---|
| `x` | `[E, 11]` | original edge features become dual node features |
| `edge_index` | `[2, D]` | face-adjacency connectivity |
| `y` | `[E]` | seam labels per dual node |

<details>
<summary>Click to expand: Building the dataset</summary>

```bash
python ./preprocessing/obj_to_dataset_graph.py [./meshes (YOUR_DIR_NAME)] --max-meshes 200 --save
```

Scans `./meshes` for `.obj` files, converts each to a PyG `Data` object with exact OBJ seam labels, prints per-mesh statistics and class balance, then saves the full list to `dataset.pt`.

Meshes with zero detected seam edges are flagged as outliers and excluded.

Feature groups:
- `paper14` is the default and builds GraphSeam-style baseline features: endpoint `[normalized xyz, normals, gaussian curvature]` for both endpoints. With `--endpoint-order auto`, this preset uses random endpoint order.
- `custom` enables optional feature engineering via `--enable-ao`, `--enable-dihedral`, `--enable-symmetry`, `--enable-density`, and `--enable-thickness-sdf` flags.

Seam detection works on actual UV data when present — an edge is a seam if either endpoint has different UV coordinates across its two adjacent faces. Boundary edges are always seams. Falls back to boundary-only detection when the mesh has no UVs.

</details>

<details>
<summary>Click to expand: Dataset audit</summary>

Audit raw `.obj` files before building a dataset:

```bash
python tools/audit_dataset.py ./3d-objs --json-out audit_raw.json --csv-out audit_raw.csv
```

Audit a serialized dataset:

```bash
python tools/audit_dataset.py dataset.pt --json-out audit_dataset.json --csv-out audit_dataset.csv
python tools/audit_dataset.py dataset_dual.pt --json-out audit_dual.json --csv-out audit_dual.csv
```

The audit prints a short console summary and writes a JSON report plus a CSV table with one row per mesh. It infers family IDs, resolution tags, augmentation status, edge/seam counts, merge statistics when raw geometry is available, and possible train/val/test leakage using the configured split ratios.

Family parsing strips augmentation suffixes such as `_aug0` and common resolution suffixes such as `_10000f` or `_res12`. Custom suffix rules can be passed with `--augmentation-pattern` and repeated `--resolution-pattern` flags.
All train/val/test splitting uses family-level grouping so augmented and resolution-varied meshes from the same source stay in one split.

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

### SparseMeshCNN dataset (`dataset_meshcnn_full_*.pt`)

SparseMeshCNN uses a separate `MeshCNNSample` dataset format and training path in `models/meshcnn_full/train.py`. The official builder is `preprocessing/build_meshcnn_dataset_v2.py`.

Paper-equivalent dataset:

```bash
python preprocessing/build_meshcnn_dataset.py ./meshes \
  --output dataset_meshcnn_full_paper14.pt \
  --feature-group paper14
```

Ablation superset dataset for runtime slicing:

```bash
python preprocessing/build_meshcnn_dataset.py ./meshes \
  --output dataset_meshcnn_full_custom_superset_random.pt \
  --feature-group custom \
  --enable-ao \
  --enable-dihedral \
  --enable-symmetry \
  --enable-density \
  --enable-thickness-sdf \
  --endpoint-order random
```

The builder writes `feature_names`, `feature_group`, `feature_preset`, `feature_flags`, `endpoint_order`, `label_source='exact_obj'`, `density_config` when present, and matching `edge_features` tensors for runtime slicing.

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
- Optional: `pyembree` (faster AO raycasting; trimesh ray_triangle is used as fallback)
