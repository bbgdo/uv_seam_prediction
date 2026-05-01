# Preprocessing

The maintained dataset builders are:

- `obj_to_dataset_graph.py` for the PyG dual-graph dataset used by GraphSAGE and GATv2
- `build_meshcnn_dataset_v2.py` for the SparseMeshCNN dataset

## GNN / PyG Dataset

```bash
python obj_to_dataset_graph.py ./3d-objs \
  --feature-group paper14 \
  --endpoint-order random \
  --save \
  --output dataset_paper14_dual.pt
```

For engineered features, build a custom superset dataset:

```bash
python obj_to_dataset_graph.py ./3d-objs \
  --feature-group custom \
  --enable-ao \
  --enable-dihedral \
  --enable-symmetry \
  --enable-density \
  --enable-thickness-sdf \
  --endpoint-order random \
  --save \
  --output dataset_custom_dual.pt
```

`paper14` and `custom` are the maintained feature groups. Label extraction uses exact OBJ seam truth.

## SparseMeshCNN Dataset

```bash
python build_meshcnn_dataset_v2.py ./3d-objs \
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

Use one custom superset dataset for SparseMeshCNN ablations and let `models/meshcnn_full/train.py` slice features at runtime.

## Dataset Audit

```bash
python ../tools/audit_dataset.py ./3d-objs --json-out audit_raw.json --csv-out audit_raw.csv
python ../tools/audit_dataset.py ../dataset_paper14_dual.pt --json-out audit_paper14.json --csv-out audit_paper14.csv
python ../tools/audit_dataset.py ../dataset_custom_dual.pt --json-out audit_custom.json --csv-out audit_custom.csv
```

`tools/audit_dataset.py` reports family IDs, resolution tags, augmentation status, seam ratios, and simulated split leakage. Leakage checks use family grouping only.
