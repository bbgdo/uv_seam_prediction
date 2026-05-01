# UV Seam Prediction

PyTorch tooling for UV seam prediction on triangulated OBJ meshes. The maintained training surface is:

- `preprocessing/obj_to_dataset_graph.py` for the PyG dual-graph dataset used by GraphSAGE and GATv2
- `preprocessing/build_meshcnn_dataset_v2.py` for the SparseMeshCNN dataset
- `tools/run_baseline.py` for single-run GraphSAGE or GATv2 training
- `tools/run_feature_ablations.py` for cross-model ablations over GraphSAGE, GATv2, and SparseMeshCNN
- `models/meshcnn_full/train.py` for direct SparseMeshCNN training
- `tools/predict_seams.py` for inference and Blender integration

## Official Entry Points

### Build datasets

```bash
python preprocessing/obj_to_dataset_graph.py ./3d-objs \
  --feature-group paper14 \
  --endpoint-order random \
  --save \
  --output dataset_paper14_dual.pt

python preprocessing/obj_to_dataset_graph.py ./3d-objs \
  --feature-group custom \
  --enable-ao \
  --enable-dihedral \
  --enable-symmetry \
  --enable-density \
  --enable-thickness-sdf \
  --endpoint-order random \
  --save \
  --output dataset_custom_dual.pt

python preprocessing/build_meshcnn_dataset_v2.py ./3d-objs \
  --feature-group custom \
  --enable-ao \
  --enable-dihedral \
  --enable-symmetry \
  --enable-density \
  --enable-thickness-sdf \
  --endpoint-order random \
  --save \
  --output dataset_sparsemeshcnn_custom.pt
```

`exact_obj` is the maintained label source. Split generation and loading use family grouping only.

### Train one GNN baseline

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
  --enable-density \
  --run-dir runs/gatv2_custom
```

### Run ablations

```bash
python tools/run_feature_ablations.py \
  --model graphsage \
  --custom-dataset dataset_custom_dual.pt \
  --experiments control14 ao_density ao_dihedral_symmetry ao_dihedral_symmetry_density_sdf \
  --seeds 7 11 19 \
  --epochs 100 \
  --output-root runs/ablations_graphsage \
  --generate-splits
```

Use `--model sparsemeshcnn` with `--meshcnn-dataset` for SparseMeshCNN ablations.

### Train SparseMeshCNN directly

```bash
python models/meshcnn_full/train.py \
  --dataset dataset_sparsemeshcnn_custom.pt \
  --feature-group custom \
  --run-dir runs/sparsemeshcnn_custom
```

### Predict seams

```bash
python tools/predict_seams.py \
  --mesh-path mesh.obj \
  --model-weights runs/graphsage_paper14/best_model.pth \
  --output-json prediction.json \
  --feature-bundle paper14
```

Public inference model names are `graphsage`, `gatv2`, and `sparsemeshcnn`.

## Utilities

- `tools/audit_dataset.py`: inspect raw OBJ directories or serialized datasets and check family-split leakage
- `tools/validate_seam_truth.py`: parity-check exact OBJ seam extraction
- `tools/evaluate_dir_topology.py`: bulk topology and post-processing evaluation over a directory of meshes
- `tools/evaluate_saved_models.py`: offline reevaluation of saved checkpoints with exact validation-threshold search

## Internal Or Archived Tools

- `tools/run_graphseam_baseline.py`: internal multi-seed paper-protocol batch wrapper around `tools/run_baseline.py`
- `evaluation/run_evaluation.py`: deprecated internal UV-study script kept for archival unwrap experiments

For more detail, see [preprocessing/README.md](preprocessing/README.md), [models/README.md](models/README.md), and [evaluation/README.md](evaluation/README.md).
