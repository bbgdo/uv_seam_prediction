# UV Seam Prediction

PyTorch tooling for UV seam prediction on triangulated OBJ meshes.

Maintained entrypoints:

- `preprocessing/build_gnn_dataset.py` for GraphSAGE/GATv2 PyG datasets
- `preprocessing/build_meshcnn_dataset.py` for SparseMeshCNN datasets
- `tools/run_training.py` for single GraphSAGE, GATv2, and SparseMeshCNN training
- `tools/run_feature_ablations.py` for GraphSAGE, GATv2, and SparseMeshCNN ablations
- `tools/predict_seams.py` for inference
- `tools/evaluate_dir_topology.py` for topology and post-processing evaluation
- `tools/evaluate_saved_models.py` for reevaluating saved checkpoints
- `tools/audit_dataset.py` for dataset audit and family split leakage inspection
- `tools/validate_seam_truth.py` for exact OBJ seam truth validation

## Reproducibility Workflow

Validate exact OBJ seam truth before building datasets:

```bash
python tools/validate_seam_truth.py --mesh-dir data/objs
```

Build the maintained PyG datasets:

```bash
python preprocessing/build_gnn_dataset.py data/objs --feature-group paper14 --endpoint-order fixed --save --output datasets/gnn_paper14.pt
python preprocessing/build_gnn_dataset.py data/objs --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --endpoint-order fixed --save --output datasets/gnn_custom.pt
```

Build one SparseMeshCNN custom superset dataset for all SparseMeshCNN ablations:

```bash
python preprocessing/build_meshcnn_dataset.py data/objs --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --endpoint-order fixed --output datasets/sparsemeshcnn_custom_superset.pt --overwrite
```

Audit dataset contents and family split leakage:

```bash
python tools/audit_dataset.py data/objs --json-out outputs/audit_raw.json --csv-out outputs/audit_raw.csv
python tools/audit_dataset.py datasets/gnn_custom.pt --json-out outputs/audit_gnn_custom.json --csv-out outputs/audit_gnn_custom.csv
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
python tools/run_feature_ablations.py --model gatv2 --gnn-dataset datasets/gnn_custom.pt --full-suite --seeds 7 11 19 --epochs 100 --output-root runs/ablations/gatv2 --generate-splits
```

Run SparseMeshCNN ablations on the single custom superset dataset:

```bash
python tools/run_feature_ablations.py --model sparsemeshcnn --meshcnn-dataset datasets/sparsemeshcnn_custom_superset.pt --full-suite --seeds 7 11 19 --epochs 100 --output-root runs/ablations/sparsemeshcnn --generate-splits
```

Run inference with the maintained bridge:

```bash
python tools/predict_seams.py --mesh-path data/objs/example.obj --model-weights runs/models/graphsage_paper14/best_model.pth --feature-bundle paper14 --output-json outputs/predictions/example.json
```

Run topology and post-processing evaluation:

```bash
python tools/evaluate_dir_topology.py --input-dir data/objs --model-weights runs/models/graphsage_paper14/best_model.pth --feature-bundle paper14 --csv-out outputs/predictions/topology.csv
```

Reevaluate saved checkpoints:

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
