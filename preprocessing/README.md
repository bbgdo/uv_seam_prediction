# Preprocessing

Maintained dataset builders:

- `preprocessing/obj_to_dataset_graph.py` for GraphSAGE/GATv2 PyG datasets
- `preprocessing/build_meshcnn_dataset.py` for SparseMeshCNN datasets

## GNN / PyG Builder

Use `preprocessing/obj_to_dataset_graph.py` for `paper14` and `custom` PyG datasets.

```bash
python preprocessing/obj_to_dataset_graph.py data/objs --feature-group paper14 --endpoint-order random --save --output datasets/gnn_paper14.pt
python preprocessing/obj_to_dataset_graph.py data/objs --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --endpoint-order random --save --output datasets/gnn_custom.pt
```

## SparseMeshCNN Builder

Use `preprocessing/build_meshcnn_dataset.py` for SparseMeshCNN. Build one custom superset dataset with all optional custom features enabled:

```bash
python preprocessing/build_meshcnn_dataset.py data/objs --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --endpoint-order random --output datasets/sparsemeshcnn_custom_superset.pt --overwrite
```

`tools/run_feature_ablations.py --model sparsemeshcnn` slices this superset at runtime. No per-ablation SparseMeshCNN datasets are required.

## Dataset Metadata

Serialized datasets and manifests should expose the same maintained metadata surface:

- `feature_names`
- `feature_group`
- `feature_preset`
- `feature_flags`
- `endpoint_order`
- `label_source='exact_obj'`
- family split compatibility

## Dataset Audit

Use `tools/audit_dataset.py` to inspect raw OBJ directories or serialized datasets and verify family-only split compatibility:

```bash
python tools/audit_dataset.py data/objs --json-out outputs/audit_raw.json --csv-out outputs/audit_raw.csv
python tools/audit_dataset.py datasets/gnn_custom.pt --json-out outputs/audit_gnn_custom.json --csv-out outputs/audit_gnn_custom.csv
```
