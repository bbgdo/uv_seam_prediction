# Models

GNN models and training utilities for UV seam edge classification.

> #### TODO:
> - [ ] New metrics based on "A Dataset and Benchmark for Mesh Parameterization" paper
> - [ ] MeshCNN architecture

---

## Architecture 1 — `graphsage/model.py`

<details>
<summary>Click to expand: Architecture details</summary>

**Forward pass:**
1. Three `SAGEConv` layers encode node features into hidden embeddings. LayerNorm and ReLU after each layer; residual connections from layer 2 onward.
2. For each directed edge `(i->j)`, build the representation: `[h_i || h_j || edge_attr]`.
3. A 3-layer MLP maps this concatenation to a single raw logit.

The edge MLP runs in chunks (`chunk_size=100_000`) to avoid OOM on large meshes.

**Default hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `node_in_dim` | 6 | xyz + normals |
| `edge_in_dim` | 11 | 11-dim feature vector |
| `hidden_dim` | 128 | SAGEConv output dim |
| `dropout` | 0.3 | applied after each conv and in MLP |

</details>

---

## Architecture 2 — `gatv2/model.py`

`DualGATv2` is a GATv2-based node classifier on the **dual graph**. Edge classification is reframed as node classification: each original edge becomes a dual node with the 11-dim feature vector as node features.

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

## Architecture 3 — `dual_graphsage/model.py`

`DualGraphSAGE` is a GraphSAGE-based node classifier on the **dual graph** — the same data as GATv2 but with SAGEConv aggregation. Enables a fair architecture comparison that isolates the model effect from the graph representation effect.

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

## Training

All training scripts share the same structure: `BCEWithLogitsLoss` with `pos_weight`, AdamW optimizer, `ReduceLROnPlateau` scheduler, early stopping on val F1, and integrated experiment logging.

An optional **connectivity penalty** (`--lambda-conn`) can be added to the loss during training. It penalizes dual-graph nodes (= original edges) whose predicted seam probability is high but whose dual-graph neighbors have low probability — discouraging isolated, topologically useless seam predictions.

### GraphSAGE (original graph) — `graphsage/train.py`

```bash
python models/graphsage/train.py \
    --dataset dataset.pt \
    --run-dir runs/graphsage_001 \
    --epochs 100 --hidden 128
```

### GATv2 (dual graph) — `gatv2/train.py`

```bash
python models/gatv2/train.py \
    --dataset dataset_dual.pt \
    --run-dir runs/gatv2_001 \
    --epochs 100 --hidden 64 --heads 8 --lr 5e-4
```

### DualGraphSAGE (dual graph) — `dual_graphsage/train.py`

```bash
python models/dual_graphsage/train.py \
    --dataset dataset_dual.pt \
    --run-dir runs/dual_graphsage_001 \
    --epochs 100 --hidden 128
```

<details>
<summary>Click to expand: Training configuration</summary>

| Setting | GATv2 | DualGraphSAGE |
|---|---|---|
| Loss | `BCEWithLogitsLoss` + `pos_weight` | same |
| Optimizer | AdamW (lr=5e-4, wd=1e-4) | AdamW (lr=1e-3, wd=1e-4) |
| LR Scheduler | `ReduceLROnPlateau` (factor=0.5, patience=5) | same |
| Early stopping | patience=15 on val F1 | same |
| Data split | 75/15/10 (seed=42) | same |
| Input dataset | `dataset_dual.pt` | `dataset_dual.pt` |

</details>

<details>
<summary>Click to expand: Full CLI options</summary>

| Flag | GATv2 | DualGraphSAGE | Description |
|---|---|---|---|
| `--dataset` | `dataset_dual.pt` | `dataset_dual.pt` | path to dataset |
| `--run-dir` | `runs/gatv2_{ts}` | `runs/dual_graphsage_{ts}` | experiment output dir |
| `--epochs` | 100 | 100 | max training epochs |
| `--lr` | 5e-4 | 1e-3 | learning rate |
| `--hidden` | 64 | 128 | hidden dim |
| `--dropout` | 0.3 | 0.3 | dropout rate |
| `--patience` | 15 | 15 | early-stop patience |
| `--lambda-conn` | 0.0 | 0.0 | connectivity penalty weight (try 0.1) |
| `--heads` | 8 | — | attention heads (GATv2 only) |

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
| `connectivity_penalty(logits, edge_index)` | Penalizes isolated seam predictions: high-prob dual nodes with low-prob neighbors |
| `seam_loss_with_connectivity(logits, labels, edge_index, pos_weight, lambda_conn)` | `BCEWithLogitsLoss` + weighted connectivity penalty |

### `postprocess.py`

Inference-time post-processing for seam predictions. Can also be run as a standalone script.

| Function | Description |
|---|---|
| `threshold_and_clean(probs, unique_edges, threshold, min_component_size)` | Threshold + remove disconnected components smaller than `min_component_size` |
| `stitch_seam_gaps(probs, seam_mask, unique_edges, max_gap)` | Greedy gap stitching: bridge gaps between seam components within `max_gap` steps |
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

`ExperimentLogger` — writes per-epoch metrics to JSON, generates training plots as PNG. See root README for output format.

### `comparison.py`

Generates cross-experiment comparison plots from multiple run directories:

```bash
python models/utils/comparison.py runs/graphsage_001 runs/gatv2_001 runs/dual_graphsage_001
```

Outputs `comparison_f1.png` (overlaid val F1 curves) and `comparison_table.png` (test results table).
