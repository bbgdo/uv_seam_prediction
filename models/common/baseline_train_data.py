from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch_geometric.data import Data

from preprocessing.feature_registry import ResolvedFeatureSet, resolve_feature_selection


METADATA_KEYS = (
    'label_source',
    'feature_group',
    'feature_names',
    'feature_flags',
    'density_config',
    'endpoint_order',
    'weld_mode',
)


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metadata_value(data: Data, key: str):
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
        if isinstance(container, dict) and key in container and container[key] not in (None, ''):
            return container[key]
        if hasattr(container, key):
            value = getattr(container, key)
            if value not in (None, ''):
                return value
    return None


def dataset_metadata_summary(dataset: list[Data]) -> dict:
    summary: dict = {'graph_count': len(dataset)}
    for key in METADATA_KEYS:
        values = []
        missing = 0
        for data in dataset:
            value = metadata_value(data, key)
            if value is None:
                missing += 1
            else:
                values.append(str(value))

        if values:
            unique_values = sorted(set(values))
            summary[key] = unique_values[0] if len(unique_values) == 1 else unique_values
        if missing and (values or missing != len(dataset)):
            summary[f'{key}_missing'] = missing

    feature_dims = []
    for data in dataset:
        x = getattr(data, 'x', None)
        if x is not None and getattr(x, 'ndim', 0) == 2:
            feature_dims.append(int(x.shape[1]))
    if feature_dims:
        unique_dims = sorted(set(feature_dims))
        summary['x_feature_dim'] = unique_dims[0] if len(unique_dims) == 1 else unique_dims

    return summary


def resolve_runtime_feature_selection(args: argparse.Namespace) -> ResolvedFeatureSet:
    feature_group = getattr(args, 'feature_group', None)
    if feature_group is None:
        feature_group = 'paper14'

    return resolve_feature_selection(
        feature_group,
        enable_ao=bool(getattr(args, 'enable_ao', False)),
        enable_dihedral=bool(getattr(args, 'enable_dihedral', False)),
        enable_symmetry=bool(getattr(args, 'enable_symmetry', False)),
        enable_density=bool(getattr(args, 'enable_density', False)),
        enable_thickness_sdf=bool(getattr(args, 'enable_thickness_sdf', False)),
    )


def coerce_feature_names(value) -> list[str] | None:
    if value in (None, ''):
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def apply_runtime_feature_selection(dataset: list[Data], selection: ResolvedFeatureSet) -> list[Data]:
    requested = list(selection.feature_names)
    for graph_idx, data in enumerate(dataset):
        feature_names = coerce_feature_names(metadata_value(data, 'feature_names'))

        current_dim = int(data.x.shape[1])
        if feature_names is None:
            raise ValueError(
                f"dataset graph {graph_idx} is missing feature_names metadata; "
                f"cannot select requested features {requested}"
            )
        if len(feature_names) != current_dim:
            raise ValueError(
                f"dataset graph {graph_idx} feature_names length {len(feature_names)} "
                f"does not match x feature dim {current_dim}"
            )

        missing = [name for name in requested if name not in feature_names]
        if missing:
            raise ValueError(
                f"dataset graph {graph_idx} is missing requested feature(s): {missing}; "
                f"available feature_names={feature_names}"
            )

        if feature_names == requested:
            continue

        indices = [feature_names.index(name) for name in requested]
        data.x = data.x[:, indices]
        data.feature_names = requested
        data.feature_group = selection.feature_group
        data.feature_flags = selection.feature_flags.as_dict()
        if selection.density_config is not None:
            data.density_config = dict(selection.density_config)

    return dataset
