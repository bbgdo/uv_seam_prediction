from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.utils.json_io import read_json


METRIC_KEYS = ('precision', 'recall', 'f1', 'fpr', 'tpr', 'accuracy')
DELTA_KEYS = ('fpr', 'recall', 'f1', 'accuracy')
PAIRED_DELTA_KEYS = METRIC_KEYS
DEFAULT_REPORT_GRID = tuple([round(value / 100, 2) for value in range(90, 100)] + [0.995, 0.999])
REEVAL_FILENAME = 'reeval_exact_threshold.json'
AGGREGATE_FILENAME = 'reeval_aggregate.json'
TIE_BREAKING = [
    'maximize validation f1',
    'minimize validation fpr',
    'maximize validation precision',
    'maximize threshold',
]


@dataclass(frozen=True)
class SavedRun:
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    summary_path: Path
    split_path: Path
    dataset_path: Path
    experiment: str | None
    seed: int
    config: dict[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True)
class ReferenceControlSet:
    reference_control_dir: Path
    experiment_name: str
    by_seed: dict[int, dict[str, Any]]


def normalize_metric_payload(metrics: dict[str, Any] | None) -> dict[str, float] | None:
    if not metrics:
        return None
    return {key: float(metrics[key]) for key in METRIC_KEYS if metrics.get(key) is not None}


def metric_delta(new_metrics: dict[str, Any] | None, old_metrics: dict[str, Any] | None) -> dict[str, float | None]:
    new_payload = normalize_metric_payload(new_metrics)
    old_payload = normalize_metric_payload(old_metrics)
    return {
        metric: (
            float(new_payload[metric]) - float(old_payload[metric])
            if new_payload is not None
            and old_payload is not None
            and new_payload.get(metric) is not None
            and old_payload.get(metric) is not None
            else None
        )
        for metric in DELTA_KEYS
    }


def canonicalize_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonicalize_for_hash(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        normalized = [canonicalize_for_hash(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(',', ':')),
        )
    return value


def split_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    payload = read_json(path)
    canonical = json.dumps(canonicalize_for_hash(payload), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def resolve_existing_split_path(raw_path: str | None, *, reeval_path: Path | None = None) -> Path | None:
    if not raw_path:
        return None
    split_path = Path(raw_path)
    candidates = [split_path]
    if not split_path.is_absolute():
        candidates.append(Path.cwd() / split_path)
        if reeval_path is not None:
            candidates.append(reeval_path.parent / split_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def split_identity(row: dict[str, Any], *, reeval_path: Path | None = None) -> dict[str, Any]:
    existing = row.get('split_identity') if isinstance(row.get('split_identity'), dict) else {}
    raw_path = row.get('split_path') or existing.get('split_path')
    seed = int(row['run_identity']['seed'])
    split_path = resolve_existing_split_path(raw_path, reeval_path=reeval_path)
    basename = Path(raw_path).name if raw_path else None
    fingerprint = existing.get('fingerprint')
    if fingerprint is None and split_path is not None:
        fingerprint = split_fingerprint(split_path)
    return {
        'seed': seed,
        'split_path': str(raw_path) if raw_path else None,
        'split_path_basename': basename,
        'fingerprint': fingerprint,
    }


def split_identities_match(target: dict[str, Any], control: dict[str, Any]) -> tuple[bool, str, str]:
    if int(target['seed']) != int(control['seed']):
        return False, 'seed_mismatch', 'seed'
    if target.get('fingerprint') and control.get('fingerprint'):
        if target['fingerprint'] == control['fingerprint']:
            return True, 'matched', 'split_content_fingerprint'
        return False, 'split_content_fingerprint_mismatch', 'split_content_fingerprint'
    if (
        target.get('split_path_basename')
        and control.get('split_path_basename')
        and target['split_path_basename'] == control['split_path_basename']
    ):
        return True, 'matched', 'split_path_basename_plus_seed'
    return False, 'split_identity_unavailable_or_mismatch', 'split_path_basename_plus_seed'


def exact_test_metrics(row: dict[str, Any], *, source: str) -> dict[str, float]:
    try:
        metrics = row['metrics']['test']['exact_val_best']
    except KeyError as exc:
        raise ValueError(f'{source} is missing metrics.test.exact_val_best') from exc
    missing = [metric for metric in PAIRED_DELTA_KEYS if metrics.get(metric) is None]
    if missing:
        raise ValueError(f'{source} missing exact-threshold test metric(s): {", ".join(missing)}')
    return {metric: float(metrics[metric]) for metric in PAIRED_DELTA_KEYS}
