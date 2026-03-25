# Models

GNN models and training utilities for UV seam edge classification.

Three architectures are provided for comparison. All classify the same 11-dim artistic edge features; the difference is in how they aggregate neighborhood information.

---

## Architecture 1 — `dual_graphsage/model.py`

`DualGraphSAGE` is a GraphSAGE-based node classifier on the **dual graph**. Each original mesh edge becomes a dual node with the 11-dim feature vector as its features. Edges in the dual graph connect dual nodes whose corresponding mesh edges share a face.

<details>
<summary>Click to expand: Architecture details</summary>

**Forward pass:**
1. Three `SAGEConv` layers with LayerNorm and ReLU. Residual connections from layer 2 onward.
2. A 2-layer classifier MLP maps node embeddings to a seam logit.

**Default hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `in_dim` | 11 | dual node features = original edge features |
| `hidden_dim` | 128 | SAGEConv output dim |
| `num_layers` | 3 | SAGEConv layers |
| `dropout` | 0.3 | applied between layers |

</details>

---

## Architecture 2 — `gatv2/model.py`

`DualGATv2` is a GATv2-based node classifier on the **dual graph** — same representation as DualGraphSAGE but with multi-head attention aggregation instead of mean pooling.

<details>
<summary>Click to expand: Architecture details</summary>

**Forward pass:**
1. Three `GATv2Conv` layers with multi-head attention (8 heads). LayerNorm and ELU after each layer.
2. Residual connections for all layers where dimensions match (middle layers onward).
3. Final layer uses a single attention head to reduce dimensionality.
4. A 2-layer classifier MLP maps node embeddings to a seam logit.

**Default hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `in_dim` | 11 | dual node features = original edge features |
| `hidden_dim` | 64 | per-head output dim |
| `heads` | 8 | attention heads (effective hidden = 64 × 8 = 512) |
| `num_layers` | 3 | GATv2Conv layers |
| `dropout` | 0.3 | applied in attention and between layers |

</details>

---

## Architecture 3 — `meshcnn/model.py`

`MeshCNNClassifier` uses the fixed 4-neighbor convolution from [MeshCNN (Hanocka et al., 2019)](https://arxiv.org/abs/1809.05910). Unlike the dual-graph models, it operates directly on the original mesh topology — no dual graph conversion. Each edge has exactly 4 neighboring edges (2 per incident triangle), forming a fixed-size convolution kernel.

This architecture serves as the third comparison point for the diploma: all three models see the same 11-dim features, isolating the architecture effect.

> **Simplification:** The standard MeshCNN includes edge-collapse pooling for mesh simplification. This implementation omits pooling and classifies at original mesh resolution, which is sufficient for seam prediction.

<details>
<summary>Click to expand: Architecture details</summary>

**MeshConv operator** (`mesh_conv.py`):
1. For each edge `(a,b)`, gather features from its 4 neighbors: `[e_ac, e_cb]` from face `(a,b,c)` and `[e_bd, e_da]` from face `(a,b,d)`.
2. Sort each pair element-wise to resolve edge direction ambiguity (symmetric aggregation).
3. Concatenate self + pair1_sorted + pair2_sorted → `[E, 5C]` → Linear → `[E, out_channels]`.

**MeshCNNClassifier forward pass:**
1. Four `MeshConv` layers with LayerNorm, ReLU, dropout, and residual connections from layer 2 onward.
2. A 2-layer classifier MLP maps edge embeddings to a seam logit.

**Default hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `in_channels` | 11 | edge feature dim |
| `hidden_channels` | 64 | MeshConv output dim (effective width = 5×64=320 inside each conv) |
| `num_layers` | 4 | MeshConv layers |
| `dropout` | 0.3 | applied between layers |

**Data preprocessing:** Run `preprocessing/build_meshcnn_data.py` to add `edge_neighbors [E, 4]` to the dataset before training.

</details>

---

## Training

All training scripts share the same structure: `BCEWithLogitsLoss` with `pos_weight`, AdamW optimizer, `ReduceLROnPlateau` scheduler, early stopping on val F1, and integrated experiment logging.

An optional **connectivity penalty** (`--lambda-conn`) penalizes isolated seam predictions — dual-graph nodes (or MeshCNN edges) whose predicted seam probability is high but whose neighbors have low probability.

### DualGraphSAGE — `dual_graphsage/train.py`

```bash
python models/dual_graphsage/train.py \
    --dataset dataset_dual.pt \
    --run-dir runs/dual_graphsage_001 \
    --epochs 100 --hidden 128
```

### DualGATv2 — `gatv2/train.py`

```bash
python models/gatv2/train.py \
    --dataset dataset_dual.pt \
    --run-dir runs/gatv2_001 \
    --epochs 100 --hidden 64 --heads 8 --lr 5e-4
```

### MeshCNN — `meshcnn/train.py`

```bash
# Step 1: build MeshCNN dataset (adds 4-neighbor indices)
python preprocessing/build_meshcnn_data.py --input dataset.pt --output dataset_meshcnn.pt

# Step 2: train
python models/meshcnn/train.py \
    --dataset dataset_meshcnn.pt \
    --run-dir runs/meshcnn_001 \
    --epochs 100 --hidden 64 --num-layers 4
```

<details>
<summary>Click to expand: Training configuration</summary>

| Setting | DualGraphSAGE | DualGATv2 | MeshCNN |
|---|---|---|---|
| Loss | `BCEWithLogitsLoss` + `pos_weight` | same | same |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) | AdamW (lr=5e-4, wd=1e-4) | AdamW (lr=1e-3, wd=1e-4) |
| LR Scheduler | `ReduceLROnPlateau` (factor=0.5, patience=5) | same | same |
| Early stopping | patience=15 on val F1 | same | same |
| Data split | 75/15/10 (seed=42) | same | same |
| Input dataset | `dataset_dual.pt` | `dataset_dual.pt` | `dataset_meshcnn.pt` |

</details>

<details>
<summary>Click to expand: Full CLI options</summary>

| Flag | DualGraphSAGE | DualGATv2 | MeshCNN | Description |
|---|---|---|---|---|
| `--dataset` | `dataset_dual.pt` | `dataset_dual.pt` | `dataset_meshcnn.pt` | path to dataset |
| `--run-dir` | `runs/dual_graphsage_{ts}` | `runs/gatv2_{ts}` | `runs/meshcnn_{ts}` | experiment output dir |
| `--epochs` | 100 | 100 | 100 | max training epochs |
| `--lr` | 1e-3 | 5e-4 | 1e-3 | learning rate |
| `--hidden` | 128 | 64 | 64 | hidden dim |
| `--dropout` | 0.3 | 0.3 | 0.3 | dropout rate |
| `--patience` | 15 | 15 | 15 | early-stop patience |
| `--lambda-conn` | 0.0 | 0.0 | 0.0 | connectivity penalty weight |
| `--heads` | — | 8 | — | attention heads (GATv2 only) |
| `--num-layers` | — | — | 4 | MeshConv layers (MeshCNN only) |
| `--in-channels` | — | — | auto | feature dim (auto-detected) |

</details>

Each run produces `config.json`, `metrics.json`, `summary.json`, training plots (`.png`), and `best_model.pth` in the `--run-dir`.

---

## Utilities — `utils/`

### `dataset.py`

| Function | Description |
|---|---|
| `load_dataset(path)` | Load `.pt` file as a list of PyG `Data` objects |
| `load_dual_dataset(path)` | Load original dataset and convert each graph to dual on-the-fly |
| `split_dataset(dataset, val_ratio, test_ratio, seed)` | Reproducible train/val/test split |
| `compute_pos_weight(dataset)` | Compute `pos_weight` tensor for `BCEWithLogitsLoss` from train set class balance |

### `metrics.py`

| Function | Description |
|---|---|
| `edge_f1(logits, labels, threshold)` | Returns `{f1, precision, recall, accuracy}` for binary edge classification |

### `losses.py`

| Function | Description |
|---|---|
| `connectivity_penalty(logits, edge_index)` | Penalizes isolated seam predictions: high-prob nodes with low-prob neighbors |
| `seam_loss_with_connectivity(logits, labels, edge_index, pos_weight, lambda_conn)` | `BCEWithLogitsLoss` + weighted connectivity penalty |

### `postprocess.py`

Inference-time post-processing for seam predictions.

| Function | Description |
|---|---|
| `threshold_and_clean(probs, unique_edges, threshold, min_component_size)` | Threshold + remove disconnected components smaller than `min_component_size` |
| `stitch_seam_gaps(probs, seam_mask, unique_edges, max_gap)` | Greedy gap stitching: bridge gaps between seam components |
| `postprocess_seams(probs, unique_edges, edge_to_faces, threshold, min_component_size, max_gap)` | Combined pipeline: threshold → clean → stitch |

```bash
python models/utils/postprocess.py \
    --dataset dataset.pt \
    --dual-dataset dataset_dual.pt \
    --weights runs/dual_graphsage_001/best_model.pth \
    --model-type graphsage \
    --threshold 0.5 --min-component 3 --max-gap 3
```

### `experiment_log.py`

`ExperimentLogger` — writes per-epoch metrics to JSON, generates training plots as PNG.

### `comparison.py`

Generates cross-experiment comparison plots from multiple run directories:

```bash
python models/utils/comparison.py runs/dual_graphsage_001 runs/gatv2_001 runs/meshcnn_001
```

Outputs `comparison_f1.png` (overlaid val F1 curves) and `comparison_table.png` (test results table).
