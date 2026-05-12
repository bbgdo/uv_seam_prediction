# Models

Maintained training entrypoints:

- `tools/run_training.py` for single GraphSAGE, GATv2, and SparseMeshCNN runs
- `tools/run_feature_ablations.py` for GraphSAGE, GATv2, and SparseMeshCNN ablations

## Single Training Runs

Train any maintained architecture through `tools/run_training.py`.

```bash
python tools/run_training.py --model graphsage --dataset datasets/gnn_paper14.pt --feature-group paper14 --run-dir runs/models/graphsage_paper14
python tools/run_training.py --model gatv2 --dataset datasets/gnn_custom.pt --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --run-dir runs/models/gatv2_custom
python tools/run_training.py --model sparsemeshcnn --dataset datasets/sparsemeshcnn_custom_superset.pt --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --run-dir runs/models/sparsemeshcnn_custom
```

`paper14` is the paper feature baseline. `custom` means `paper14` plus at least one optional feature toggle.

## SparseMeshCNN

Run SparseMeshCNN ablations through `tools/run_feature_ablations.py`.

```bash
python tools/run_feature_ablations.py --model sparsemeshcnn --meshcnn-dataset datasets/sparsemeshcnn_custom_superset.pt --seeds 7 11 19 --epochs 100 --output-root runs/ablations/sparsemeshcnn --generate-splits
```

The maintained model name is `sparsemeshcnn`. `models/meshcnn_full/` is only the internal module path.

## Ablation Protocol

- `paper14` is the paper baseline bundle.
- `custom` is the runtime-selectable superset built on top of `paper14`.
- Optional custom features are `ao`, `dihedral`, `symmetry`, `density`, and `sdf`.
- Split protocol is `family` only.
- No connectivity loss is used.
