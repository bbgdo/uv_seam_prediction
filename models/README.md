# Models

GNN model and training utilities for UV seam edge classification.

> **Note:** The current architecture and training script are an MVP and will be rewritten. **It is more like a placeholder, not a final design.**
>
> #### TODO:
> - [ ] New metrics based on "A Dataset and Benchmark for Mesh Parameterization" paper.
> - [ ] features like simmetry, placement of seams in crease or on sharp edges, etc.
> - [ ] GAT, MPNN, MeshCNN
---

## Architecture — `graphsage/model.py`

`UVSeamGNN` is a GraphSAGE-based edge classifier that returns raw logits (use `BCEWithLogitsLoss` during training).

<details>
<summary>Click to expand: Architecture details</summary>

**Forward pass:**
1. Two `SAGEConv` layers encode node features into hidden embeddings `h`.
2. For each directed edge `(i→j)`, build the representation: `[h_i || h_j || edge_attr]`.
3. A 3-layer MLP maps this concatenation to a single raw logit.

The edge MLP runs in chunks (`chunk_size=100_000`) to avoid OOM on large meshes — materialising the full `[E, 260]` tensor at once can exceed VRAM for high-poly models.

**Default hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `node_in_dim` | 6 | xyz + normals |
| `edge_in_dim` | 4 | length, dihedral, Δnorm, dot_norm |
| `hidden_dim` | 128 | SAGEConv output dim |
| `dropout` | 0.3 | applied after each conv and in MLP |

</details>

---

## Training — `graphsage/train.py`

<details>
<summary>Click to expand: Training configuration</summary>

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` with `pos_weight = non_seam / seam` |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | `ReduceLROnPlateau` — halves LR if val F1 stagnates for 5 epochs |
| Early stopping | patience = 15 epochs on val F1 |
| Data split | 75% train / 15% val / 10% test (seeded shuffle, seed=42) |

Graphs exceeding `--max-edges` are skipped to avoid OOM during training. Best checkpoint is saved to `best_model.pth` based on validation F1.

</details>

```bash
python models/graphsage/train.py \
    --dataset dataset.pt \
    --save-dir models/graphsage \
    --epochs 100 \
    --hidden 128
```

**Full CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `dataset.pt` | path to dataset |
| `--save-dir` | `models/graphsage` | where to write `best_model.pth` |
| `--epochs` | 100 | max training epochs |
| `--lr` | 1e-3 | learning rate |
| `--hidden` | 128 | hidden dim |
| `--dropout` | 0.3 | dropout rate |
| `--patience` | 15 | early-stop patience |
| `--max-edges` | 2,000,000 | skip graphs above this edge count |

---

## Utilities — `utils/`

### `dataset.py`

| Function | Description |
|---|---|
| `load_dataset(path)` | Load `.pt` file as a list of PyG `Data` objects |
| `split_dataset(dataset, val_ratio, test_ratio, seed)` | Reproducible train/val/test split |
| `compute_pos_weight(dataset)` | Compute `pos_weight` tensor for `BCEWithLogitsLoss` from train set class balance |

### `metrics.py`

| Function | Description |
|---|---|
| `edge_f1(logits, labels, threshold)` | Returns `{f1, precision, recall, accuracy}` for binary edge classification |
