# UV Seam Prediction

PyTorch tooling for UV seam prediction on triangulated OBJ meshes.

Maintained entrypoints:

- `preprocessing/build_gnn_dataset.py` for GraphSAGE/GATv2 PyG datasets
- `preprocessing/build_meshcnn_dataset.py` for SparseMeshCNN datasets
- `tools/run_training.py` for single GraphSAGE, GATv2, and SparseMeshCNN training
- `tools/run_feature_ablations.py` for GraphSAGE, GATv2, and SparseMeshCNN ablations
- `tools/predict_seams.py` for inference
- `tools/evaluate_saved_models.py` for reevaluating saved GraphSAGE/GATv2 checkpoints

## Reproducibility Workflow

Build the maintained PyG datasets:

```bash
python preprocessing/build_gnn_dataset.py data/objs --feature-group paper14 --endpoint-order fixed --save --output datasets/gnn_paper14.pt
python preprocessing/build_gnn_dataset.py data/objs --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --endpoint-order fixed --save --output datasets/gnn_custom.pt
```

Build one SparseMeshCNN custom superset dataset for all SparseMeshCNN ablations:

```bash
python preprocessing/build_meshcnn_dataset.py data/objs --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --endpoint-order fixed --output datasets/sparsemeshcnn_custom_superset.pt --overwrite
```

Run a single training job:

```bash
python tools/run_training.py --model graphsage --dataset datasets/gnn_paper14.pt --feature-group paper14 --run-dir runs/models/graphsage_paper14
python tools/run_training.py --model gatv2 --dataset datasets/gnn_custom.pt --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --run-dir runs/models/gatv2_custom
python tools/run_training.py --model sparsemeshcnn --dataset datasets/sparsemeshcnn_custom_superset.pt --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --run-dir runs/models/sparsemeshcnn_custom
```

Run GraphSAGE and GATv2 feature ablations on the custom superset dataset:

```bash
python tools/run_feature_ablations.py --model graphsage --gnn-dataset datasets/gnn_custom.pt --experiments control14 ao density ao_dihedral_symmetry_density_sdf --seeds 7 11 19 --epochs 100 --output-root runs/ablations/graphsage --generate-splits
python tools/run_feature_ablations.py --model gatv2 --gnn-dataset datasets/gnn_custom.pt --seeds 7 11 19 --epochs 100 --output-root runs/ablations/gatv2 --generate-splits
```

Run SparseMeshCNN ablations on the single custom superset dataset:

```bash
python tools/run_feature_ablations.py --model sparsemeshcnn --meshcnn-dataset datasets/sparsemeshcnn_custom_superset.pt --seeds 7 11 19 --epochs 100 --output-root runs/ablations/sparsemeshcnn --generate-splits
```

Run inference with the maintained bridge:

```bash
python tools/predict_seams.py --mesh-path data/objs/example.obj --model-weights runs/models/graphsage_paper14/best_model.pth --feature-bundle paper14 --threshold 0.5 --output-json outputs/predictions/example.json
```

## Artifact Compatibility

The maintained CLI names are `graphsage`, `gatv2`, and `sparsemeshcnn`. Prediction requires `best_model.pth`, adjacent `config.json`, and an explicit `--threshold`.

Prediction accepts these legacy artifact metadata shapes:

- SparseMeshCNN configs with old model names `meshcnn_full`, `meshcnn`, or `sparse_meshcnn`; outputs still report `sparsemeshcnn`.
- GraphSAGE configs with `aggr: mean`; new GraphSAGE training still defaults to LSTM aggregation.
- GNN configs that stored hidden width as `hidden` or `hidden_size` instead of `hidden_dim`.
- Checkpoints wrapped as `state_dict` or `model_state_dict`, plus current raw state dicts and `model_state`.

These compatibility paths are for artifact loading only. New commands and docs should use the maintained names and metadata.

Reevaluate saved GraphSAGE/GATv2 checkpoints:

```bash
python tools/evaluate_saved_models.py --runs-root runs/ablations/graphsage --splits-dir runs/ablations/graphsage/splits --gnn-dataset datasets/gnn_custom.pt
```

Notes:

- Label source is `exact_obj`.
- Split protocol is `family` only.
- Canonical feature groups are `paper14` and `custom`.
- `paper14` is the base bundle. `custom` means `paper14` plus at least one optional feature toggle.
- `control14` is an ablation experiment name, not a third feature group.
- SparseMeshCNN is the maintained model name in CLIs, configs, and outputs. `models/meshcnn_full/` is only the module path.

See [preprocessing/README.md](preprocessing/README.md) and [models/README.md](models/README.md) for the maintained command surface.
