import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.baselines.registry import get_baseline  # noqa: E402
from models.common.baseline_train import (  # noqa: E402
    _collect_logits_labels,
    _model_kwargs,
    apply_runtime_feature_selection,
)
from models.common.config import baseline_config, replace_config  # noqa: E402
from models.utils.dataset import filter_dataset_by_resolution, load_dataset, load_split_json_metadata, split_dataset  # noqa: E402
from models.utils.metrics import binary_metrics_from_probs  # noqa: E402
from preprocessing.feature_registry import resolve_feature_selection  # noqa: E402
from tools.run_feature_ablations import EXPERIMENT_SPECS, split_path_for_seed  # noqa: E402


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


def _read_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _normalize_metric_payload(metrics: dict[str, Any] | None) -> dict[str, float] | None:
    if not metrics:
        return None
    return {key: float(metrics[key]) for key in METRIC_KEYS if metrics.get(key) is not None}


def _metric_delta(new_metrics: dict[str, Any] | None, old_metrics: dict[str, Any] | None) -> dict[str, float | None]:
    new_payload = _normalize_metric_payload(new_metrics)
    old_payload = _normalize_metric_payload(old_metrics)
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


def _canonicalize_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize_for_hash(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        normalized = [_canonicalize_for_hash(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(',', ':')),
        )
    return value


def _split_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    canonical = json.dumps(_canonicalize_for_hash(payload), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _resolve_existing_split_path(raw_path: str | None, *, reeval_path: Path | None = None) -> Path | None:
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


def _split_identity(row: dict[str, Any], *, reeval_path: Path | None = None) -> dict[str, Any]:
    existing = row.get('split_identity') if isinstance(row.get('split_identity'), dict) else {}
    raw_path = row.get('split_path') or existing.get('split_path')
    seed = int(row['run_identity']['seed'])
    split_path = _resolve_existing_split_path(raw_path, reeval_path=reeval_path)
    basename = Path(raw_path).name if raw_path else None
    fingerprint = existing.get('fingerprint')
    if fingerprint is None and split_path is not None:
        fingerprint = _split_fingerprint(split_path)
    return {
        'seed': seed,
        'split_path': str(raw_path) if raw_path else None,
        'split_path_basename': basename,
        'fingerprint': fingerprint,
    }


def _split_identities_match(target: dict[str, Any], control: dict[str, Any]) -> tuple[bool, str, str]:
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


def _exact_test_metrics(row: dict[str, Any], *, source: str) -> dict[str, float]:
    try:
        metrics = row['metrics']['test']['exact_val_best']
    except KeyError as exc:
        raise ValueError(f'{source} is missing metrics.test.exact_val_best') from exc
    missing = [metric for metric in PAIRED_DELTA_KEYS if metrics.get(metric) is None]
    if missing:
        raise ValueError(f'{source} missing exact-threshold test metric(s): {", ".join(missing)}')
    return {metric: float(metrics[metric]) for metric in PAIRED_DELTA_KEYS}


def load_reference_control_reevaluations(reference_control_dir: str | Path) -> ReferenceControlSet:
    root = Path(reference_control_dir)
    if not root.exists():
        raise ValueError(f'--reference-control-dir does not exist: {root}')

    by_seed: dict[int, dict[str, Any]] = {}
    experiment_names = []
    for reeval_path in sorted(root.rglob(REEVAL_FILENAME)):
        payload = _read_json(reeval_path)
        if not isinstance(payload, dict) or payload.get('status') != 'completed':
            continue
        if 'run_identity' not in payload or payload['run_identity'].get('seed') is None:
            raise ValueError(f'{reeval_path} is missing run_identity.seed')
        seed = int(payload['run_identity']['seed'])
        if seed in by_seed:
            raise ValueError(f'duplicate reference control reevaluation for seed {seed}: {reeval_path}')
        _exact_test_metrics(payload, source=str(reeval_path))
        by_seed[seed] = {
            'payload': payload,
            'reeval_path': str(reeval_path),
            'split_identity': _split_identity(payload, reeval_path=reeval_path),
        }
        experiment = payload['run_identity'].get('experiment')
        if experiment:
            experiment_names.append(str(experiment))

    if not by_seed:
        raise ValueError(f'no authoritative {REEVAL_FILENAME} files found under {root}')

    unique_names = sorted(set(experiment_names))
    experiment_name = unique_names[0] if len(unique_names) == 1 else root.name
    return ReferenceControlSet(
        reference_control_dir=root,
        experiment_name=experiment_name,
        by_seed=by_seed,
    )


def build_report_grid(spec: str | None = None) -> list[float]:
    if spec is None or not spec.strip():
        values = list(DEFAULT_REPORT_GRID)
    elif ':' in spec and ',' not in spec:
        parts = [float(part) for part in spec.split(':')]
        if len(parts) != 3:
            raise ValueError('--report-grid range must be start:stop:step')
        start, stop, step = parts
        if step <= 0:
            raise ValueError('--report-grid step must be positive')
        values = []
        current = start
        while current <= stop + (step / 2):
            values.append(round(current, 12))
            current += step
    else:
        values = [float(part.strip()) for part in spec.split(',') if part.strip()]

    cleaned = []
    for value in values:
        if value < 0 or value > 1:
            raise ValueError(f'threshold must be between 0 and 1, got {value}')
        if math.isclose(value, 1.0):
            continue
        cleaned.append(float(value))

    grid = sorted(set(cleaned))
    if not grid:
        raise ValueError('report grid is empty after excluding threshold 1.0')
    return grid


def _metrics_at_threshold(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, Any]:
    metrics = binary_metrics_from_probs(probs, labels, threshold=float(threshold))
    metrics['threshold'] = float(threshold)
    return metrics


def threshold_table(probs: torch.Tensor, labels: torch.Tensor, thresholds: list[float]) -> list[dict[str, Any]]:
    return [_metrics_at_threshold(probs, labels, threshold) for threshold in thresholds]


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _best_threshold_index(
    *,
    f1: np.ndarray,
    fpr: np.ndarray,
    precision: np.ndarray,
    thresholds: np.ndarray,
) -> int:
    order = np.lexsort((thresholds, precision, -fpr, f1))
    return int(order[-1])


def compute_threshold_metrics_fast(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    prob_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    label_arr = np.asarray(labels).reshape(-1)
    if prob_arr.size == 0:
        raise ValueError('cannot search threshold on an empty validation set')
    if prob_arr.size != label_arr.size:
        raise ValueError('probabilities and labels must have the same number of elements')

    labels_bool = label_arr.astype(bool)
    order = np.argsort(-prob_arr, kind='mergesort')
    sorted_probs = prob_arr[order]
    sorted_labels = labels_bool[order]

    group_ends = np.flatnonzero(np.r_[sorted_probs[1:] != sorted_probs[:-1], True])
    thresholds = sorted_probs[group_ends]
    valid = thresholds < 1.0
    thresholds = thresholds[valid]
    group_ends = group_ends[valid]

    cumulative_pos = np.cumsum(sorted_labels, dtype=np.int64)
    tp = cumulative_pos[group_ends] if group_ends.size else np.array([], dtype=np.int64)
    predicted_positive = group_ends.astype(np.int64) + 1
    fp = predicted_positive - tp

    max_score = float(np.max(prob_arr))
    above_max = float(np.nextafter(np.float64(max_score), np.float64(1.0)))
    if max_score < 1.0 and above_max < 1.0:
        thresholds = np.r_[above_max, thresholds]
        tp = np.r_[np.int64(0), tp]
        fp = np.r_[np.int64(0), fp]
    elif thresholds.size == 0:
        fallback_threshold = float(np.nextafter(np.float64(1.0), np.float64(0.0)))
        positive_at_fallback = prob_arr >= fallback_threshold
        thresholds = np.array([fallback_threshold], dtype=np.float64)
        tp = np.array([np.count_nonzero(positive_at_fallback & labels_bool)], dtype=np.int64)
        fp = np.array([np.count_nonzero(positive_at_fallback & ~labels_bool)], dtype=np.int64)

    total = int(prob_arr.size)
    total_pos = int(np.count_nonzero(labels_bool))
    total_neg = total - total_pos
    fn = total_pos - tp
    tn = total_neg - fp

    precision = _safe_divide(tp.astype(np.float64), (tp + fp).astype(np.float64))
    recall = _safe_divide(tp.astype(np.float64), (tp + fn).astype(np.float64))
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-8)
    fpr = _safe_divide(fp.astype(np.float64), (fp + tn).astype(np.float64))
    accuracy = (tp + tn).astype(np.float64) / max(total, 1)

    best_index = _best_threshold_index(
        f1=f1,
        fpr=fpr,
        precision=precision,
        thresholds=thresholds,
    )
    best_metrics = {
        'f1': float(f1[best_index]),
        'precision': float(precision[best_index]),
        'recall': float(recall[best_index]),
        'accuracy': float(accuracy[best_index]),
        'fpr': float(fpr[best_index]),
        'tpr': float(recall[best_index]),
        'tp': int(tp[best_index]),
        'fp': int(fp[best_index]),
        'fn': int(fn[best_index]),
        'tn': int(tn[best_index]),
        'threshold': float(thresholds[best_index]),
    }

    return {
        'threshold': float(thresholds[best_index]),
        'metrics': best_metrics,
        'candidate_count': int(thresholds.size),
        'candidate_source': (
            'unique validation score breakpoints below 1.0 using cumulative counts; '
            'includes nextafter(max_score, 1.0) no-positive breakpoint when below 1.0'
        ),
        'tie_breaking': list(TIE_BREAKING),
    }


def exact_validation_threshold(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    flat_probs = probs.detach().flatten().cpu().numpy()
    flat_labels = labels.detach().flatten().cpu().numpy()
    return compute_threshold_metrics_fast(flat_probs, flat_labels)


def _infer_experiment(run_dir: Path, runs_root: Path) -> str | None:
    try:
        parts = run_dir.resolve().relative_to(runs_root.resolve()).parts
    except ValueError:
        parts = run_dir.parts
    if 'experiments' in parts:
        index = parts.index('experiments')
        if index + 1 < len(parts):
            return parts[index + 1]
    if run_dir.parent.name in EXPERIMENT_SPECS:
        return run_dir.parent.name
    return None


def _parse_seed(run_dir: Path, config: dict[str, Any], summary: dict[str, Any]) -> int:
    for source in (config, summary):
        value = source.get('seed')
        if value is not None:
            return int(value)
    if run_dir.name.startswith('seed_'):
        return int(run_dir.name.split('_', 1)[1])
    raise ValueError(f'{run_dir} is missing seed metadata')


def _experiment_summary(run_dir: Path, experiment: str | None) -> dict[str, Any]:
    if experiment is None:
        return {}
    summary_path = run_dir.parent / 'summary.json'
    if not summary_path.exists():
        return {}
    payload = _read_json(summary_path)
    return payload if isinstance(payload, dict) else {}


def _select_dataset_path(
    *,
    config: dict[str, Any],
    experiment_summary: dict[str, Any],
    custom_dataset: str | None,
) -> Path:
    if custom_dataset:
        return Path(custom_dataset)
    dataset = config.get('dataset') or experiment_summary.get('dataset')
    if not dataset:
        raise ValueError('missing dataset path metadata; provide --custom-dataset')
    return Path(dataset)


def _require_config(config: dict[str, Any], run_dir: Path) -> None:
    required = ['model_name', 'in_dim', 'hidden_dim', 'num_layers', 'dropout', 'dataset', 'seed']
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f'{run_dir / "config.json"} missing required field(s): {", ".join(missing)}')
    if config.get('model_name') == 'graphsage':
        for key in ('aggr', 'skip_connections'):
            if config.get(key) is None:
                raise ValueError(f'{run_dir / "config.json"} missing required field: {key}')


def discover_saved_runs(args: argparse.Namespace) -> list[SavedRun]:
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise ValueError(f'--runs-root does not exist: {runs_root}')
    splits_dir = Path(args.splits_dir)
    targets: list[SavedRun] = []
    requested_experiments = set(args.experiments or [])
    requested_seeds = {int(seed) for seed in args.seeds} if args.seeds else None

    for checkpoint_path in sorted(runs_root.rglob('best_model.pth')):
        run_dir = checkpoint_path.parent
        experiment = _infer_experiment(run_dir, runs_root)
        if requested_experiments and experiment not in requested_experiments:
            continue

        config_path = run_dir / 'config.json'
        summary_path = run_dir / 'summary.json'
        if not config_path.exists():
            raise ValueError(f'{run_dir} has best_model.pth but no config.json')
        if not summary_path.exists():
            raise ValueError(f'{run_dir} has best_model.pth but no summary.json')

        config = _read_json(config_path)
        summary = _read_json(summary_path)
        if not isinstance(config, dict) or not isinstance(summary, dict):
            raise ValueError(f'{run_dir} config.json and summary.json must contain objects')
        _require_config(config, run_dir)

        seed = _parse_seed(run_dir, config, summary)
        if requested_seeds is not None and seed not in requested_seeds:
            continue

        experiment_summary = _experiment_summary(run_dir, experiment)
        split_path = split_path_for_seed(splits_dir, seed)
        if not split_path.exists():
            raise ValueError(f'missing frozen split JSON for seed {seed}: {split_path}')
        dataset_path = _select_dataset_path(
            config=config,
            experiment_summary=experiment_summary,
            custom_dataset=args.custom_dataset,
        )

        targets.append(SavedRun(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            summary_path=summary_path,
            split_path=split_path,
            dataset_path=dataset_path,
            experiment=experiment,
            seed=seed,
            config=config,
            summary=summary,
        ))

    if not targets:
        raise ValueError('no matching runs with best_model.pth were found')
    return targets


def _feature_selection_from_config(config: dict[str, Any]):
    flags = config.get('feature_flags') or {}
    return resolve_feature_selection(
        config.get('feature_group'),
        enable_ao=bool(flags.get('ao', False)),
        enable_signed_dihedral=bool(flags.get('signed_dihedral', flags.get('dihedral', False))),
        enable_symmetry=bool(flags.get('symmetry', False)),
        enable_density=bool(flags.get('density', False)),
    )


def _runtime_config_from_saved(config: dict[str, Any]):
    definition = get_baseline(config['model_name'])
    base = baseline_config(config['model_name'], definition.default_config_overrides)
    return replace_config(
        base,
        hidden_size=config.get('hidden_dim'),
        num_layers=config.get('num_layers'),
        in_dim=config.get('in_dim'),
        dropout=config.get('dropout'),
        lr=config.get('lr'),
        pos_weight=config.get('pos_weight'),
        focal_gamma=config.get('focal_gamma'),
        patience=config.get('patience'),
        heads=config.get('heads'),
        aggr=config.get('aggr'),
        skip_connections=config.get('skip_connections'),
    )


def _load_state_dict(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    if not isinstance(checkpoint, dict):
        raise ValueError(f'checkpoint must contain a state dict: {path}')
    return checkpoint


def _old_validation_best_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / 'val_threshold_sweep.json'
    if not path.exists():
        return None
    payload = _read_json(path)
    if isinstance(payload, dict):
        return payload.get('best')
    return None


def evaluate_saved_run(target: SavedRun, *, device: torch.device, report_grid: list[float]) -> dict[str, Any]:
    config = target.config
    selection = _feature_selection_from_config(config)
    runtime_config = _runtime_config_from_saved(config)
    definition = get_baseline(runtime_config.model_name)

    dataset = load_dataset(target.dataset_path)
    resolution_tag = config.get('resolution_tag', 'all')
    dataset = filter_dataset_by_resolution(dataset, resolution_tag)
    dataset = apply_runtime_feature_selection(dataset, selection)

    split_payload = load_split_json_metadata(target.split_path)
    split_dataset_path = split_payload.get('dataset_path') or None
    _, val, test, split_info = split_dataset(
        dataset,
        seed=target.seed,
        split_json_in=target.split_path,
        dataset_path=split_dataset_path,
        resolution_tag=resolution_tag,
    )

    model = definition.model_class(**_model_kwargs(runtime_config)).to(device)
    model.load_state_dict(_load_state_dict(target.checkpoint_path, device))
    model.eval()

    val_logits, val_labels = _collect_logits_labels(model, val, device)
    test_logits, test_labels = _collect_logits_labels(model, test, device)
    val_probs = torch.sigmoid(val_logits)
    test_probs = torch.sigmoid(test_logits)

    exact = exact_validation_threshold(val_probs, val_labels)
    exact_threshold = float(exact['threshold'])
    old_validation_best = _old_validation_best_metrics(target.run_dir)
    old_threshold = target.summary.get('best_validation_threshold')
    if old_threshold is None and old_validation_best:
        old_threshold = old_validation_best.get('threshold')

    val_metrics = {
        'threshold_0_5': _metrics_at_threshold(val_probs, val_labels, 0.5),
        'exact_val_best': exact['metrics'],
    }
    test_metrics = {
        'threshold_0_5': _metrics_at_threshold(test_probs, test_labels, 0.5),
        'exact_val_best': _metrics_at_threshold(test_probs, test_labels, exact_threshold),
    }
    if old_threshold is not None:
        old_threshold = float(old_threshold)
        val_metrics['old_val_best'] = _metrics_at_threshold(val_probs, val_labels, old_threshold)
        test_metrics['old_val_best'] = _metrics_at_threshold(test_probs, test_labels, old_threshold)
    else:
        val_metrics['old_val_best'] = None
        test_metrics['old_val_best'] = None

    old_stored = {
        'validation_val_best': old_validation_best,
        'test_val_best': target.summary.get('test_metrics_best_validation_threshold'),
        'test_0_5': target.summary.get('test_metrics_threshold_0_5'),
    }

    payload = {
        'status': 'completed',
        'run_identity': {
            'experiment': target.experiment,
            'seed': target.seed,
            'run_dir': str(target.run_dir),
        },
        'checkpoint_path': str(target.checkpoint_path),
        'split_path': str(target.split_path),
        'dataset_path': str(target.dataset_path),
        'model_family': {
            'model_name': runtime_config.model_name,
            'display_name': definition.display_name,
        },
        'feature_selection': {
            'feature_group': selection.feature_group,
            'feature_preset': selection.feature_preset,
            'feature_flags': selection.feature_flags.as_dict(),
            'feature_names': list(selection.feature_names),
        },
        'split': {
            'seed': split_info.get('seed'),
            'group_mode': split_info.get('group_mode'),
            'resolution_tag': split_info.get('resolution_tag'),
            'val_graphs': len(val),
            'test_graphs': len(test),
        },
        'threshold_search': {
            'method': 'exact_validation_f1_over_score_breakpoints',
            'candidate_source': exact['candidate_source'],
            'candidate_count': exact['candidate_count'],
            'tie_breaking': exact['tie_breaking'],
            'dense_report_grid': report_grid,
        },
        'old_threshold': old_threshold,
        'exact_validation_optimal_threshold': exact_threshold,
        'metrics': {
            'validation': val_metrics,
            'test': test_metrics,
        },
        'dense_grid': {
            'validation': threshold_table(val_probs, val_labels, report_grid),
            'test': threshold_table(test_probs, test_labels, report_grid),
        },
        'old_stored_metrics': old_stored,
        'comparison': {
            'delta_vs_old_stored_val_best': {
                'validation': _metric_delta(val_metrics['exact_val_best'], old_stored['validation_val_best']),
                'test': _metric_delta(test_metrics['exact_val_best'], old_stored['test_val_best']),
            },
            'delta_vs_0_5': {
                'validation': _metric_delta(val_metrics['exact_val_best'], val_metrics['threshold_0_5']),
                'test': _metric_delta(test_metrics['exact_val_best'], test_metrics['threshold_0_5']),
            },
        },
    }
    payload['split_identity'] = _split_identity(payload)
    return payload


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    return {
        'mean': statistics.mean(values) if values else None,
        'std': statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        'n': len(values),
    }


def _summarize_metric_block(rows: list[dict[str, Any]], metric_path: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRIC_KEYS:
        values = []
        for row in rows:
            cursor: Any = row
            for key in metric_path:
                cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
            if isinstance(cursor, dict) and cursor.get(metric) is not None:
                values.append(float(cursor[metric]))
        summary[metric] = _mean_std(values)
    return summary


def _summarize_delta_block(rows: list[dict[str, Any]], delta_path: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in DELTA_KEYS:
        values = []
        for row in rows:
            cursor: Any = row
            for key in delta_path:
                cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
            if isinstance(cursor, dict) and cursor.get(metric) is not None:
                values.append(float(cursor[metric]))
        summary[metric] = _mean_std(values)
    return summary


def _paired_delta_vs_control(
    experiment: str,
    rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_seed = {int(row['run_identity']['seed']): row for row in rows}
    control_by_seed = {int(row['run_identity']['seed']): row for row in control_rows}
    paired_seeds = sorted(set(by_seed) & set(control_by_seed))
    per_seed = []
    for seed in paired_seeds:
        row = {'seed': seed}
        metrics = by_seed[seed]['metrics']['test']['exact_val_best']
        control_metrics = control_by_seed[seed]['metrics']['test']['exact_val_best']
        for metric in PAIRED_DELTA_KEYS:
            row[metric] = float(metrics[metric]) - float(control_metrics[metric])
        per_seed.append(row)
    return {
        'experiment': experiment,
        'control': 'control14',
        'paired_seed_count': len(per_seed),
        'paired_seeds': paired_seeds,
        'per_seed': per_seed,
        'summary': {metric: _mean_std([float(row[metric]) for row in per_seed]) for metric in PAIRED_DELTA_KEYS},
    }


def _paired_delta_vs_reference_control(
    experiment: str,
    rows: list[dict[str, Any]],
    reference_control: ReferenceControlSet,
) -> dict[str, Any]:
    by_seed = {int(row['run_identity']['seed']): row for row in rows}
    per_seed = []
    skipped = []

    for seed in sorted(by_seed):
        target = by_seed[seed]
        control = reference_control.by_seed.get(seed)
        if control is None:
            skipped.append({'seed': seed, 'reason': 'missing_reference_control_seed'})
            continue

        target_identity = _split_identity(target)
        control_identity = control['split_identity']
        matched, reason, method = _split_identities_match(target_identity, control_identity)
        if not matched:
            skipped.append({
                'seed': seed,
                'reason': reason,
                'identity_check': method,
                'target_split_identity': target_identity,
                'control_split_identity': control_identity,
            })
            continue

        metrics = _exact_test_metrics(target, source=f'{experiment} seed {seed}')
        control_metrics = _exact_test_metrics(
            control['payload'],
            source=f'{reference_control.experiment_name} seed {seed}',
        )
        row = {
            'seed': seed,
            'identity_check': method,
        }
        for metric in PAIRED_DELTA_KEYS:
            row[metric] = float(metrics[metric]) - float(control_metrics[metric])
        per_seed.append(row)

    if not per_seed:
        raise ValueError(
            f'no valid external control pairings remain for experiment {experiment}; '
            f'skipped seeds: {skipped}'
        )

    return {
        'experiment': experiment,
        'control': reference_control.experiment_name,
        'paired_seed_count': len(per_seed),
        'paired_seeds': [int(row['seed']) for row in per_seed],
        'skipped_seeds': skipped,
        'per_seed': per_seed,
        'summary': {metric: _mean_std([float(row[metric]) for row in per_seed]) for metric in PAIRED_DELTA_KEYS},
    }


def aggregate_reevaluations(
    results: list[dict[str, Any]],
    *,
    reference_control: ReferenceControlSet | None = None,
) -> dict[str, Any]:
    completed = [row for row in results if row.get('status') == 'completed']
    experiments = sorted({row['run_identity'].get('experiment') or 'unscoped' for row in completed})
    by_experiment = {
        experiment: [
            row for row in completed
            if (row['run_identity'].get('experiment') or 'unscoped') == experiment
        ]
        for experiment in experiments
    }

    experiment_summaries = {}
    for experiment, rows in by_experiment.items():
        experiment_summaries[experiment] = {
            'run_count': len(rows),
            'seeds': sorted(int(row['run_identity']['seed']) for row in rows),
            'test_exact_threshold': _summarize_metric_block(rows, ('metrics', 'test', 'exact_val_best')),
            'test_threshold_0_5': _summarize_metric_block(rows, ('metrics', 'test', 'threshold_0_5')),
            'paired_delta_vs_old_stored_val_best': _summarize_delta_block(
                rows,
                ('comparison', 'delta_vs_old_stored_val_best', 'test'),
            ),
        }

    paired_vs_control = {}
    control_rows = by_experiment.get('control14')
    if control_rows:
        for experiment, rows in by_experiment.items():
            if experiment == 'control14':
                continue
            paired_vs_control[experiment] = _paired_delta_vs_control(experiment, rows, control_rows)

    payload = {
        'run_count': len(completed),
        'experiments': experiment_summaries,
        'paired_delta_vs_control14': paired_vs_control,
    }

    if reference_control is not None:
        paired_vs_reference = {}
        for experiment, rows in by_experiment.items():
            if experiment == reference_control.experiment_name:
                continue
            paired_vs_reference[experiment] = _paired_delta_vs_reference_control(
                experiment,
                rows,
                reference_control,
            )
        if not paired_vs_reference and completed:
            raise ValueError('no target experiments remain for external reference control pairing')
        payload['external_reference_control'] = {
            'reference_control_dir': str(reference_control.reference_control_dir),
            'reference_experiment_name': reference_control.experiment_name,
            'seeds_loaded': sorted(reference_control.by_seed),
        }
        payload['paired_delta_vs_reference_control'] = paired_vs_reference

    return payload


def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('--device cuda requested but CUDA is not available')
    return torch.device(name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Offline reevaluate saved best_model.pth checkpoints.')
    parser.add_argument('--runs-root', required=True, help='root directory containing experiment outputs')
    parser.add_argument('--splits-dir', required=True, help='directory containing frozen seed split JSON files')
    parser.add_argument('--custom-dataset', default=None, help='custom/superset dual dataset override')
    parser.add_argument('--experiments', nargs='+', default=None, help='experiment names to reevaluate')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='seed numbers to reevaluate')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--report-grid', default=None, help='comma list or start:stop:step threshold grid')
    parser.add_argument('--reference-control-dir', default=None, help='previously reevaluated control experiment dir')
    parser.add_argument('--write-json', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dry-run', action='store_true', help='show matching runs without running inference')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report_grid = build_report_grid(args.report_grid)
        targets = discover_saved_runs(args)
        if args.dry_run:
            for target in targets:
                print(
                    f"would evaluate: experiment={target.experiment or '-'} seed={target.seed} "
                    f"run_dir={target.run_dir}"
                )
            return

        reference_control = (
            load_reference_control_reevaluations(args.reference_control_dir)
            if args.reference_control_dir
            else None
        )
        device = resolve_device(args.device)
        results = []
        for target in targets:
            print(f"reevaluating {target.experiment or 'run'} seed {target.seed}: {target.run_dir}")
            payload = evaluate_saved_run(target, device=device, report_grid=report_grid)
            results.append(payload)
            if args.write_json:
                _write_json(target.run_dir / REEVAL_FILENAME, payload)
                print(f"  wrote {target.run_dir / REEVAL_FILENAME}")

        if args.write_json and (len(results) > 1 or reference_control is not None):
            aggregate = aggregate_reevaluations(results, reference_control=reference_control)
            output_path = Path(args.runs_root) / AGGREGATE_FILENAME
            _write_json(output_path, aggregate)
            print(f"aggregate written -> {output_path}")
            for experiment, delta in aggregate.get('paired_delta_vs_reference_control', {}).items():
                skipped = delta.get('skipped_seeds') or []
                if skipped:
                    print(
                        f"warning: external control pairing for {experiment} skipped {len(skipped)} seed(s)",
                        file=sys.stderr,
                    )
    except ValueError as exc:
        raise SystemExit(f'error: {exc}') from exc


if __name__ == '__main__':
    main()
