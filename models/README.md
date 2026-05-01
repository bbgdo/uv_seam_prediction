# Models

Maintained training entrypoints:

- `tools/run_baseline.py` for single GraphSAGE and GATv2 runs
- `tools/run_feature_ablations.py` for GraphSAGE, GATv2, and SparseMeshCNN ablations
- `models/meshcnn_full/train.py` for direct SparseMeshCNN training

## GraphSAGE and GATv2

Train GraphSAGE or GATv2 through `tools/run_baseline.py`.

```bash
python tools/run_baseline.py --model graphsage --dataset datasets/gnn_paper14.pt --feature-group paper14 --preset paper --run-dir runs/models/graphsage_paper14
python tools/run_baseline.py --model gatv2 --dataset datasets/gnn_custom.pt --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --run-dir runs/models/gatv2_custom
```

`paper14` is the paper feature baseline.

## SparseMeshCNN

Train SparseMeshCNN directly through `models/meshcnn_full/train.py` or run ablations through `tools/run_feature_ablations.py`.

```bash
python models/meshcnn_full/train.py --dataset datasets/sparsemeshcnn_custom_superset.pt --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --run-dir runs/models/sparsemeshcnn_custom
python tools/run_feature_ablations.py --model sparsemeshcnn --meshcnn-dataset datasets/sparsemeshcnn_custom_superset.pt --full-suite --seeds 7 11 19 --epochs 100 --output-root runs/ablations/sparsemeshcnn --generate-splits
```

The public model name in the ablation runner is `sparsemeshcnn`. `models/meshcnn_full/` is the internal path.

## Ablation Protocol

- `paper14` is the paper baseline bundle.
- `custom` is the runtime-selectable superset.
- Optional custom features are `ao`, `dihedral`, `symmetry`, `density`, and `sdf`.
- Split protocol is `family` only.
- No connectivity loss is used.
