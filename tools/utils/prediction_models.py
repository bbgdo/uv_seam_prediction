from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.baselines.registry import get_baseline
from models.meshcnn_full.model import MeshCNNSegmenter
from tools.utils.prediction_common import PredictionError, coerce_dict, normalize_model_name


def resolve_device(requested: str) -> torch.device:
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise PredictionError('requested CUDA device, but CUDA is unavailable', 'UnavailableDevice')
    return torch.device(requested)


def resolve_model_type(requested: str, config: dict[str, Any], weights_path: Path) -> str:
    if requested != 'auto':
        resolved = normalize_model_name(requested)
        return resolved or requested

    for key in ('model', 'model_name'):
        resolved = normalize_model_name(config.get(key))
        if resolved is not None:
            return resolved

    resolved = normalize_model_name(weights_path.parent.name)
    if resolved is not None:
        return resolved

    raise PredictionError(
        'model type could not be resolved from --model-type, config metadata, or parent run directory',
        'MissingModelType',
    )


def resolve_model_kwargs(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    if model_name == 'sparsemeshcnn':
        model_config = coerce_dict(config.get('model_config')) or {}
        feature_metadata = coerce_dict(config.get('feature_metadata')) or {}
        sources = (model_config, config, feature_metadata)
        in_channels = required_config_value_from_sources(
            sources,
            ('in_channels', 'in_dim', 'feature_dim'),
            'in_channels/in_dim/feature_dim',
        )
        hidden_channels = required_config_value_from_sources(
            sources,
            ('hidden_channels', 'hidden_size', 'hidden_dim', 'hidden'),
            'hidden_channels/hidden_size/hidden_dim/hidden',
        )
        kwargs = {
            'in_channels': int(in_channels),
            'hidden_channels': int(hidden_channels),
            'dropout': float(optional_config_value_from_sources(sources, ('dropout',), 0.2)),
            'pool_ratios': coerce_float_tuple(
                optional_config_value_from_sources(sources, ('pool_ratios',), (0.85, 0.75)),
                'pool_ratios',
            ),
            'min_edges': int(optional_config_value_from_sources(sources, ('min_edges',), 32)),
        }
        max_pool_collapses = optional_config_value_from_sources(sources, ('max_pool_collapses',), None)
        if max_pool_collapses is not None:
            kwargs['max_pool_collapses'] = int(max_pool_collapses)
        return kwargs

    in_dim = required_config_value(config, ('in_dim',), 'in_dim')
    hidden_dim = required_config_value(config, ('hidden_size', 'hidden_dim', 'hidden'), 'hidden_size/hidden_dim/hidden')
    kwargs = {
        'in_dim': int(in_dim),
        'hidden_dim': int(hidden_dim),
        'num_layers': int(required_config_value(config, ('num_layers',), 'num_layers')),
        'dropout': float(required_config_value(config, ('dropout',), 'dropout')),
    }
    if model_name == 'gatv2':
        kwargs['heads'] = int(required_config_value(config, ('heads',), 'heads'))
    elif model_name == 'graphsage':
        kwargs['aggr'] = str(required_config_value(config, ('aggr',), 'aggr'))
        kwargs['skip_connections'] = str(
            required_config_value(config, ('skip_connections',), 'skip_connections')
        )
    else:
        raise PredictionError(f'unsupported model type: {model_name}', 'InvalidModelType')
    return kwargs


def required_config_value(config: dict[str, Any], keys: tuple[str, ...], label: str) -> Any:
    for key in keys:
        if key in config and config[key] not in (None, ''):
            return config[key]
    raise PredictionError(f'config metadata is missing required model key: {label}', 'InvalidConfig')


def required_config_value_from_sources(
    sources: tuple[dict[str, Any], ...],
    keys: tuple[str, ...],
    label: str,
) -> Any:
    value = optional_config_value_from_sources(sources, keys, None)
    if value is None:
        raise PredictionError(f'config metadata is missing required model key: {label}', 'InvalidConfig')
    return value


def optional_config_value_from_sources(
    sources: tuple[dict[str, Any], ...],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value not in (None, ''):
                return value
    return default


def coerce_float_tuple(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, str):
        value = [item.strip() for item in value.split(',') if item.strip()]
    if not isinstance(value, (list, tuple)):
        raise PredictionError(f'{label} must be a list, tuple, or comma-separated string', 'InvalidConfig')
    result = tuple(float(item) for item in value)
    if not result:
        raise PredictionError(f'{label} must contain at least one value', 'InvalidConfig')
    return result


def normalize_probabilities(probabilities: np.ndarray, expected_length: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape == (expected_length,):
        return probs.astype(float)
    if probs.shape == (expected_length, 1):
        return probs[:, 0].astype(float)
    if probs.shape == (1, expected_length):
        return probs[0].astype(float)
    raise PredictionError(
        f'model output shape {probs.shape} cannot be normalized to {expected_length} edge probabilities',
        'InvalidModelOutput',
    )


def load_weights_payload(weights_path: Path, device: torch.device) -> Any:
    try:
        try:
            return torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(weights_path, map_location=device)
        except Exception:
            return torch.load(weights_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise PredictionError(f'failed to load model weights: {weights_path}', 'InvalidWeights') from exc


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ('state_dict', 'model_state_dict', 'model_state'):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return nested
        if all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise PredictionError(
        'model weights must be a state_dict or contain state_dict/model_state_dict',
        'InvalidWeights',
    )


def load_state_dict(weights_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    return extract_state_dict(load_weights_payload(weights_path, device))


def build_prediction_model(model_type: str, model_kwargs: dict[str, Any]) -> torch.nn.Module:
    if model_type == 'sparsemeshcnn':
        return MeshCNNSegmenter(**model_kwargs)
    definition = get_baseline(model_type)
    return definition.model_class(**model_kwargs)
