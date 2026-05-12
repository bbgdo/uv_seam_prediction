# Preprocessing

Maintained dataset builders:

- `preprocessing/build_gnn_dataset.py` for GraphSAGE/GATv2 PyG datasets
- `preprocessing/build_meshcnn_dataset.py` for SparseMeshCNN datasets

## GNN / PyG Builder

Use `preprocessing/build_gnn_dataset.py` for `paper14` and custom-superset PyG datasets.

```bash
python preprocessing/build_gnn_dataset.py data/objs --feature-group paper14 --endpoint-order random --save --output datasets/gnn_paper14.pt
python preprocessing/build_gnn_dataset.py data/objs --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --endpoint-order fixed --save --output datasets/gnn_custom.pt
```

`paper14` is the base bundle. `custom` is reserved for `paper14` plus at least one optional feature toggle.

## SparseMeshCNN Builder

Use `preprocessing/build_meshcnn_dataset.py` for SparseMeshCNN. Build one custom superset dataset with all optional custom features enabled:

```bash
python preprocessing/build_meshcnn_dataset.py data/objs --feature-group custom --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --endpoint-order fixed --output datasets/sparsemeshcnn_custom_superset.pt --overwrite
```

`tools/run_feature_ablations.py --model sparsemeshcnn` slices this superset at runtime. No per-ablation SparseMeshCNN datasets are required.

## Dataset Metadata

Serialized datasets and manifests should expose the same maintained metadata surface:

- `feature_names`
- `feature_group`
- `feature_flags`
- `endpoint_order`
- `label_source='exact_obj'`
- family split compatibility

## Autodesk Character Generator FBX

Use Blender to extract body meshes from Autodesk Character Generator FBX exports. The script selects meshes matching `H_DDS_(MidRes|HighRes|LowRes)`, clears parenting while keeping transforms, and exports OBJ files with UVs preserved.

```bash
blender --background --python preprocessing/autodesk_char_gen/fbx_to_obj.py -- data/fbx --out data/objs
```

If `--out` is omitted, OBJ files are written next to the FBX files.
