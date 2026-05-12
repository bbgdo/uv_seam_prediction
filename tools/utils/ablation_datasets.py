from __future__ import annotations

import argparse

from models.meshcnn_full.mesh import load_meshcnn_dataset
from models.utils.dataset import filter_dataset_by_resolution, load_dataset
from preprocessing.label_sources import EXACT_OBJ_LABEL_SOURCE

from .ablation_specs import experiment_feature_selection, is_meshcnn_model


def get_gnn_dataset_arg(args: argparse.Namespace) -> str | None:
    return getattr(args, 'gnn_dataset', None)


def _metadata_value(data, key: str):
    try:
        value = getattr(data, key)
        if value not in (None, ''):
            return value
    except AttributeError:
        pass

    for container_key in ('metadata', 'meta', 'dataset_metadata'):
        try:
            container = getattr(data, container_key)
        except AttributeError:
            continue
        if isinstance(container, dict) and container.get(key) not in (None, ''):
            return container[key]
        if hasattr(container, key):
            value = getattr(container, key)
            if value not in (None, ''):
                return value
    return None


def _unique_string_values(dataset: list, key: str) -> tuple[list[str], int]:
    values = []
    missing = 0
    for data in dataset:
        value = _metadata_value(data, key)
        if value in (None, ''):
            missing += 1
        else:
            values.append(str(value))
    return sorted(set(values)), missing


def _require_uniform_metadata(dataset: list, *, role: str, key: str, expected: str) -> None:
    observed, missing = _unique_string_values(dataset, key)
    if missing or observed != [expected]:
        detail = f"observed={observed or 'none'}"
        if missing:
            detail += f", missing={missing}"
        raise ValueError(f"{role} dataset {key} must be {expected!r} ({detail})")


def _require_uniform_metadata_choice(dataset: list, *, role: str, key: str, expected: tuple[str, ...]) -> None:
    observed, missing = _unique_string_values(dataset, key)
    if missing or len(observed) != 1 or observed[0] not in expected:
        detail = f"observed={observed or 'none'}"
        if missing:
            detail += f", missing={missing}"
        choices = ', '.join(repr(value) for value in expected)
        raise ValueError(f"{role} dataset {key} must be one of: {choices} ({detail})")


def _coerce_feature_names(value) -> list[str] | None:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def validate_gnn_dataset_metadata(dataset: list, experiment_names: list[str]) -> None:
    if not dataset:
        raise ValueError('GNN dataset is empty after resolution filtering')
    _require_uniform_metadata(dataset, role='GNN', key='feature_group', expected='custom')
    _require_uniform_metadata_choice(dataset, role='GNN', key='endpoint_order', expected=('fixed', 'random'))

    requested_features: list[str] = []
    for name in experiment_names:
        for feature_name in experiment_feature_selection(name).feature_names:
            if feature_name not in requested_features:
                requested_features.append(feature_name)

    if not requested_features:
        return

    for graph_idx, data in enumerate(dataset):
        names = _coerce_feature_names(_metadata_value(data, 'feature_names'))
        if names is None:
            raise ValueError(f"GNN dataset graph {graph_idx} is missing feature_names metadata")
        x = getattr(data, 'x', None)
        if x is not None and len(names) != int(x.shape[1]):
            raise ValueError(
                f"GNN dataset graph {graph_idx} feature_names length {len(names)} "
                f"does not match x feature dim {int(x.shape[1])}"
            )
        missing = [feature for feature in requested_features if feature not in names]
        if missing:
            raise ValueError(
                f"GNN dataset graph {graph_idx} is missing requested feature(s): {missing}; "
                f"available feature_names={names}"
            )


def validate_meshcnn_dataset_metadata(dataset: list, experiment_names: list[str]) -> None:
    if not dataset:
        raise ValueError('MeshCNN dataset is empty after resolution filtering')
    _require_uniform_metadata_choice(dataset, role='MeshCNN', key='endpoint_order', expected=('fixed', 'random'))

    label_sources, missing_label_source = _unique_string_values(dataset, 'label_source')
    if label_sources and label_sources != [EXACT_OBJ_LABEL_SOURCE]:
        detail = f"observed={label_sources}"
        if missing_label_source:
            detail += f", missing={missing_label_source}"
        raise ValueError(
            f"MeshCNN dataset label_source must be {EXACT_OBJ_LABEL_SOURCE!r} when present ({detail})"
        )

    available_names = _coerce_feature_names(_metadata_value(dataset[0], 'feature_names'))
    if available_names is None:
        raise ValueError('MeshCNN dataset sample 0 is missing feature_names metadata')

    for sample_idx, sample in enumerate(dataset):
        names = _coerce_feature_names(_metadata_value(sample, 'feature_names'))
        if names is None:
            raise ValueError(f'MeshCNN dataset sample {sample_idx} is missing feature_names metadata')
        if names != available_names:
            raise ValueError(f'MeshCNN dataset sample {sample_idx} feature_names differ from sample 0')
        edge_features = getattr(sample, 'edge_features', None)
        if edge_features is not None and int(edge_features.shape[1]) != len(names):
            raise ValueError(
                f'MeshCNN dataset sample {sample_idx} feature_names length {len(names)} '
                f'does not match edge_features dim {int(edge_features.shape[1])}'
            )

    for name in experiment_names:
        missing = [
            feature
            for feature in experiment_feature_selection(name).feature_names
            if feature not in available_names
        ]
        if missing:
            raise ValueError(
                f"MeshCNN dataset is missing requested feature(s) for {name}: {missing}; "
                f"available feature_names={available_names}"
            )


def load_filtered_dataset(path: str, resolution_tag: str) -> list:
    return filter_dataset_by_resolution(load_dataset(path), resolution_tag)


def load_filtered_meshcnn_dataset(path: str, resolution_tag: str) -> list:
    return filter_dataset_by_resolution(load_meshcnn_dataset(path), resolution_tag)


def validate_dataset_roles(args: argparse.Namespace, experiment_names: list[str]) -> dict[str, list]:
    return validate_dataset_roles_with_loaders(
        args,
        experiment_names,
        load_gnn_dataset=load_filtered_dataset,
        load_meshcnn_dataset_fn=load_filtered_meshcnn_dataset,
    )


def validate_dataset_roles_with_loaders(
    args: argparse.Namespace,
    experiment_names: list[str],
    *,
    load_gnn_dataset,
    load_meshcnn_dataset_fn,
) -> dict[str, list]:
    datasets: dict[str, list] = {}
    if is_meshcnn_model(getattr(args, 'model', 'graphsage')):
        if not args.meshcnn_dataset:
            raise ValueError('--meshcnn-dataset is required for sparsemeshcnn')
        datasets['meshcnn'] = load_meshcnn_dataset_fn(args.meshcnn_dataset, args.resolution_tag)
        validate_meshcnn_dataset_metadata(datasets['meshcnn'], experiment_names)
        return datasets

    gnn_dataset = get_gnn_dataset_arg(args)
    if not gnn_dataset:
        raise ValueError('--gnn-dataset is required for GNN models')
    datasets['gnn'] = load_gnn_dataset(gnn_dataset, args.resolution_tag)
    validate_gnn_dataset_metadata(datasets['gnn'], experiment_names)
    return datasets
