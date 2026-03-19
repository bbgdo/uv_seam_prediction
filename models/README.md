# Models

GNN models and training utilities for UV seam edge classification.

> #### TODO:
> - [ ] New metrics based on "A Dataset and Benchmark for Mesh Parameterization" paper
> - [ ] MeshCNN architecture
> - [ ] Connectivity loss for topologically valid seam loops
---

## Architecture 1 — `graphsage/model.py`

`UVSeamGNN` is a GraphSAGE-based edge classifier on the **original graph**. Returns raw logits (use `BCEWithLogitsLoss`).

<details>
<summary>Click to expand: Architecture details</summary>

**Forward pass:**
1. Three `SAGEConv` layers encode node features into hidden embeddings. The 3rd layer has a residual connection from the 2nd layer output.
2. For each directed edge `(i->j)`, build the representation: `[h_i || h_j || edge_attr]`.
3. A 3-layer MLP maps this concatenation to a single raw logit.

The edge MLP runs in chunks (`chunk_size=100_000`) to avoid OOM on large meshes.

**Default hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `node_in_dim` | 6 | xyz + normals |
| `edge_in_dim` | 11 | 11-dim feature vector (see `preprocessing/compute_features.py`) |
| `hidden_dim` | 128 | SAGEConv output dim |
| `dropout` | 0.3 | applied after each conv and in MLP |

</details>

---

## Architecture 2 — `gatv2/model.py`

`DualGATv2` is a GATv2-based node classifier on the **dual graph**. Edge classification is reframed as node classification: each original edge becomes a dual node with the 11-dim feature vector as node features.

<details>
<summary>Click to expand: Architecture details</summary>

**Forward pass:**
1. Three `GATv2Conv` layers with multi-head attention (8 heads). LayerNorm and ELU activation after each layer.
2. Residual connections for all layers where dimensions match (middle layers onward).
3. Final layer uses a single attention head to reduce dimensionality.
4. A 2-layer classifier MLP maps node embeddings to a seam logit.

**Default hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `in_dim` | 11 | dual node features = original edge features |
| `hidden_dim` | 64 | per-head output dim |
| `heads` | 8 | attention heads (effective hidden = 64 * 8 = 512) |
| `num_layers` | 3 | GATv2Conv layers |
| `dropout` | 0.3 | applied in attention and between layers |

</details>

---

## Training

Both training scripts share the same structure: `BCEWithLogitsLoss` with `pos_weight`, AdamW optimizer, `ReduceLROnPlateau` scheduler, early stopping on val F1, and integrated experiment logging.

### GraphSAGE — `graphsage/train.py`

```bash
python models/graphsage/train.py \
    --dataset dataset.pt \
    --run-dir runs/graphsage_001 \
    --epochs 100 \
    --hidden 128
```

### GATv2 — `gatv2/train.py`

```bash
python models/gatv2/train.py \
    --dataset dataset_dual.pt \
    --run-dir runs/gatv2_001 \
    --epochs 100 \
    --hidden 64 --heads 8 --lr 5e-4
```

<details>
<summary>Click to expand: Training configuration</summary>

| Setting | GraphSAGE | GATv2 |
|---|---|---|
| Loss | `BCEWithLogitsLoss` with `pos_weight` | same |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) | AdamW (lr=5e-4, wd=1e-4) |
| LR Scheduler | `ReduceLROnPlateau` (factor=0.5, patience=5) | same |
| Early stopping | patience=15 on val F1 | same |
| Data split | 75/15/10 (seed=42) | same |
| Input dataset | `dataset.pt` (original graph) | `dataset_dual.pt` (dual graph) |

</details>

<details>
<summary>Click to expand: Full CLI options (common to both)</summary>

| Flag | GraphSAGE default | GATv2 default | Description |
|---|---|---|---|
| `--dataset` | `dataset.pt` | `dataset_dual.pt` | path to dataset |
| `--run-dir` | `runs/graphsage_{timestamp}` | `runs/gatv2_{timestamp}` | experiment output dir |
| `--epochs` | 100 | 100 | max training epochs |
| `--lr` | 1e-3 | 5e-4 | learning rate |
| `--hidden` | 128 | 64 | hidden dim |
| `--dropout` | 0.3 | 0.3 | dropout rate |
| `--patience` | 15 | 15 | early-stop patience |

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

### `experiment_log.py`

`ExperimentLogger` — writes per-epoch metrics to JSON, generates training plots as PNG. See root README for output format.

### `comparison.py`

Generates cross-experiment comparison plots from multiple run directories:

```bash
python models/utils/comparison.py runs/graphsage_001 runs/gatv2_001
```

Outputs `comparison_f1.png` (overlaid val F1 curves) and `comparison_table.png` (test results table).
