from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any


MODEL_TYPES = ('auto', 'gatv2', 'graphsage', 'sparsemeshcnn')
FEATURE_BUNDLES = ('auto', 'paper14', 'custom')
_MODEL_TYPE_ALIASES: dict[str, str] = {}


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
    normalized = _MODEL_TYPE_ALIASES.get(normalized, normalized)
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
    if normalized == 'gatv2' or 'gatv2' in normalized:
        return 'gatv2'
    if normalized == 'graphsage' or 'graphsage' in normalized:
        return 'graphsage'
    if normalized in ('meshcnn_full', 'meshcnn', 'sparsemeshcnn', 'sparse_meshcnn'):
        return 'sparsemeshcnn'
    if 'meshcnn_full' in normalized or ('meshcnn' in normalized and 'sparse' in normalized):
        return 'sparsemeshcnn'
    return None


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise PredictionError(f'{label} is not valid JSON: {path}', 'InvalidJson') from exc
    if not isinstance(payload, dict):
        raise PredictionError(f'{label} must contain a JSON object: {path}', 'InvalidJson')
    return payload


def resolve_threshold(
    explicit_threshold: float | None,
    summary: dict[str, Any],
    fail_if_missing: bool = True,
) -> float:
    if explicit_threshold is not None:
        return validate_threshold(explicit_threshold)

    if 'best_validation_threshold' in summary:
        return validate_threshold(summary['best_validation_threshold'])

    suffix = ''
    if not fail_if_missing:
        suffix = '; no alternate threshold policy is implemented'
    raise PredictionError(
        'threshold is required: pass --threshold or provide summary.json["best_validation_threshold"]' + suffix,
        'MissingThreshold',
    )


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
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        value = parsed
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def coerce_dict(value: Any) -> dict[str, Any] | None:
    if value in (None, ''):
        return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        value = parsed
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return None


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PredictionError(f'{label} not found: {path}', 'MissingFile')
