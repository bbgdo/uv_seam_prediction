from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.meshcnn_full.mesh import SPARSE_MESHCNN_SAMPLE_FORMAT, MeshCNNSample
from preprocessing.feature_registry import ResolvedFeatureSet, resolve_feature_selection
from preprocessing.label_sources import EXACT_OBJ_LABEL_SOURCE


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f'{dataset_path.stem}_manifest.json')


def load_manifest(dataset_path: Path) -> dict[str, Any]:
    path = manifest_path(dataset_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def coerce_feature_names(value) -> list[str] | None:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def resolve_meshcnn_feature_selection(args) -> ResolvedFeatureSet:
    return resolve_feature_selection(
        args.feature_group,
        enable_ao=args.enable_ao,
        enable_dihedral=args.enable_dihedral,
        enable_symmetry=args.enable_symmetry,
        enable_density=args.enable_density,
        enable_thickness_sdf=args.enable_thickness_sdf,
    )


def available_meshcnn_feature_names(
    dataset: list[MeshCNNSample],
    manifest: dict[str, Any] | None,
) -> list[str]:
    if not dataset:
        raise ValueError('MeshCNN dataset is empty')

    manifest_names = coerce_feature_names((manifest or {}).get('feature_names'))
    first_names = coerce_feature_names(getattr(dataset[0], 'feature_names', None))
    available = manifest_names or first_names
    if available is None:
        raise ValueError('MeshCNN dataset is missing feature_names metadata')

    for sample_idx, sample in enumerate(dataset):
        sample_names = coerce_feature_names(getattr(sample, 'feature_names', None))
        if sample_names is None:
            raise ValueError(f'MeshCNN sample {sample_idx} is missing feature_names metadata')
        if sample_names != available:
            raise ValueError(
                f'MeshCNN sample {sample_idx} feature_names differ from the dataset feature_names metadata'
            )
        feature_dim = int(sample.edge_features.shape[1])
        if feature_dim != len(available):
            raise ValueError(
                f'MeshCNN sample {sample_idx} edge_features dim {feature_dim} does not match '
                f'feature_names length {len(available)}'
            )
    return available


def selected_feature_metadata(
    sample: MeshCNNSample,
    manifest: dict[str, Any] | None,
    selection: ResolvedFeatureSet,
    source_feature_names: list[str],
) -> dict[str, Any]:
    manifest = manifest or {}
    return {
        'feature_group': selection.feature_group,
        'feature_names': list(selection.feature_names),
        'feature_flags': selection.feature_flags.as_dict(),
        'feature_dim': len(selection.feature_names),
        'endpoint_order': manifest.get('endpoint_order', sample.endpoint_order),
        'density_config': selection.density_config,
        'label_source': manifest.get('label_source', getattr(sample, 'label_source', EXACT_OBJ_LABEL_SOURCE)),
        'sample_format': SPARSE_MESHCNN_SAMPLE_FORMAT,
        'source_feature_names': list(source_feature_names),
        'original_feature_names': list(source_feature_names),
    }


def slice_meshcnn_dataset_features(
    dataset: list[MeshCNNSample],
    selection: ResolvedFeatureSet,
    manifest: dict[str, Any] | None = None,
) -> tuple[list[MeshCNNSample], dict[str, Any]]:
    available = available_meshcnn_feature_names(dataset, manifest)
    index_by_name = {name: idx for idx, name in enumerate(available)}
    missing = [name for name in selection.feature_names if name not in index_by_name]
    if missing:
        raise ValueError(
            f"MeshCNN dataset is missing requested feature(s): {missing}; "
            f'available feature_names={available}'
        )

    selected_indices = torch.as_tensor(
        [index_by_name[name] for name in selection.feature_names],
        dtype=torch.long,
    )
    for sample_idx, sample in enumerate(dataset):
        sample.edge_features = torch.index_select(
            sample.edge_features.detach().cpu(),
            dim=1,
            index=selected_indices,
        ).contiguous()
        sample.feature_group = selection.feature_group
        sample.feature_names = list(selection.feature_names)
        sample.feature_flags = selection.feature_flags.as_dict()
        sample.density_config = dict(selection.density_config) if selection.density_config else None
        if int(sample.edge_features.shape[1]) != len(selection.feature_names):
            raise ValueError(
                f'MeshCNN sample {sample_idx} sliced feature dim {int(sample.edge_features.shape[1])} '
                f'does not match selected feature count {len(selection.feature_names)}'
            )

    return dataset, selected_feature_metadata(dataset[0], manifest, selection, available)


def validate_dataset_tensors_cpu(dataset: list[MeshCNNSample]) -> None:
    tensor_names = (
        'vertices',
        'faces',
        'unique_edges',
        'edge_features',
        'edge_labels',
    )
    for sample_idx, sample in enumerate(dataset[: min(8, len(dataset))]):
        for name in tensor_names:
            tensor = getattr(sample, name)
            if tensor.device.type != 'cpu':
                raise RuntimeError(
                    f'MeshCNNSample.{name} in sample {sample_idx} must be on CPU after dataset load, '
                    f'got {tensor.device}. Dataset loading must normalize samples to CPU.'
                )
