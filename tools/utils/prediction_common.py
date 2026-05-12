from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


MODEL_TYPES = ('auto', 'gatv2', 'graphsage', 'sparsemeshcnn')
FEATURE_BUNDLES = ('auto', 'paper14', 'custom')
MODEL_NAME_ALIASES = {
    'gatv2': 'gatv2',
    'dualgatv2': 'gatv2',
    'dual_gatv2': 'gatv2',
    'graphsage': 'graphsage',
    'dualgraphsage': 'graphsage',
    'dual_graphsage': 'graphsage',
    'sparsemeshcnn': 'sparsemeshcnn',
}
LEGACY_ARTIFACT_MODEL_NAME_ALIASES = {
    'meshcnn': 'sparsemeshcnn',
    'meshcnn_full': 'sparsemeshcnn',
    'sparse_meshcnn': 'sparsemeshcnn',
}


class PredictionError(RuntimeError):
    def __init__(self, message: str, error_type: str = 'PredictionError'):
        super().__init__(message)
        self.error_type = error_type


def normalize_feature_bundle_arg(value: str) -> str:
    normalized = str(value).strip().lower().replace('-', '_')
    if normalized not in FEATURE_BUNDLES:
        raise SystemExit(
            f"error: argument --feature-bundle: invalid choice: {value!r} "
            f"(choose from {', '.join(FEATURE_BUNDLES)})"
        )
    return normalized


def normalize_cli_model_type(value: str) -> str:
    normalized = str(value).strip().lower().replace('-', '_')
    if normalized not in MODEL_TYPES:
        raise SystemExit(
            f"error: argument --model-type: invalid choice: {value!r} "
            f"(choose from {', '.join(MODEL_TYPES)})"
        )
    return normalized


def normalize_model_name(value: Any) -> str | None:
    if value in (None, ''):
        return None
    normalized = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    return MODEL_NAME_ALIASES.get(normalized)


def normalize_artifact_model_name(value: Any) -> str | None:
    if value in (None, ''):
        return None
    normalized = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    return MODEL_NAME_ALIASES.get(normalized) or LEGACY_ARTIFACT_MODEL_NAME_ALIASES.get(normalized)


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise PredictionError(f'{label} is not valid JSON: {path}', 'InvalidJson') from exc
    if not isinstance(payload, dict):
        raise PredictionError(f'{label} must contain a JSON object: {path}', 'InvalidJson')
    return payload


def resolve_threshold(explicit_threshold: float | None) -> float:
    if explicit_threshold is None:
        raise PredictionError('threshold is required: pass --threshold', 'MissingThreshold')
    return validate_threshold(explicit_threshold)


def validate_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionError(f'threshold must be a number, got {value!r}', 'InvalidThreshold') from exc
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise PredictionError(f'threshold must be a finite value in [0, 1], got {threshold}', 'InvalidThreshold')
    return threshold


def normalize_metadata_name(value: Any) -> str | None:
    if value in (None, ''):
        return None
    return str(value).strip().lower().replace('-', '_').replace(' ', '_')


def coerce_list(value: Any) -> list[str] | None:
    if value in (None, ''):
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def coerce_dict(value: Any) -> dict[str, Any] | None:
    if value in (None, ''):
        return None
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return None


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PredictionError(f'{label} not found: {path}', 'MissingFile')
