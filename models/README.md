# Models

This repository maintains three training flows:

- `tools/run_baseline.py` for single-run GraphSAGE and GATv2 training
- `tools/run_feature_ablations.py` for multi-seed ablations across GraphSAGE, GATv2, and SparseMeshCNN
- `models/meshcnn_full/train.py` for direct SparseMeshCNN training

## Official Training Entry Points

### GraphSAGE or GATv2

```bash
python tools/run_baseline.py \
  --model graphsage \
  --dataset dataset_paper14_dual.pt \
  --feature-group paper14 \
  --run-dir runs/graphsage_paper14

python tools/run_baseline.py \
  --model gatv2 \
  --dataset dataset_custom_dual.pt \
  --feature-group custom \
  --enable-ao \
  --enable-symmetry \
  --run-dir runs/gatv2_custom
```

`--preset paper` keeps the paper-style GraphSAGE hyperparameter profile. Split grouping is `family` only.

### SparseMeshCNN

```bash
python models/meshcnn_full/train.py \
  --dataset dataset_sparsemeshcnn_custom.pt \
  --feature-group custom \
  --run-dir runs/sparsemeshcnn_custom
```

Build the corresponding dataset with `preprocessing/build_meshcnn_dataset_v2.py`.

### Ablations

```bash
python tools/run_feature_ablations.py \
  --model sparsemeshcnn \
  --meshcnn-dataset dataset_sparsemeshcnn_custom.pt \
  --full-suite \
  --seeds 7 11 19 \
  --epochs 100 \
  --output-root runs/ablations_sparsemeshcnn \
  --generate-splits
```

## Utility Scripts

- `tools/run_graphseam_baseline.py`: internal multi-seed GraphSeam paper-protocol wrapper over `tools/run_baseline.py`
- `tools/evaluate_saved_models.py`: offline reevaluation of saved checkpoints and threshold summaries

## Notes

- Public ablation model names are `graphsage`, `gatv2`, and `sparsemeshcnn`.
- Feature groups are `paper14` and `custom`.
- Maintained labels come from exact OBJ seam truth.
