from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from tools.utils.ablation_reports import (
    METRIC_KEYS,
    THRESHOLD_05_PREFIX,
    VAL_BEST_PREFIX,
    aggregate_records,
    load_existing_suite_payloads,
    write_experiment_reports,
    write_suite_reports,
)
from tools.utils.json_io import read_json, write_json
from tools.utils.reeval_common import REEVAL_FILENAME


def metric_value(metrics: dict[str, Any], metric: str) -> float | int | None:
    value = metrics.get(metric)
    if value is None:
        return None
    if metric in {'tp', 'fp', 'fn', 'tn'}:
        return int(value)
    return float(value)


def metric_columns(prefix: str, metrics: dict[str, Any]) -> dict[str, float | int | None]:
    return {f'{prefix}_{metric}': metric_value(metrics, metric) for metric in METRIC_KEYS}


def confusion_counts(metrics: dict[str, Any]) -> dict[str, int]:
    return {key: int(metrics[key]) for key in ('tp', 'fp', 'fn', 'tn') if metrics.get(key) is not None}


def load_reevaluation_rows(
    runs_root: Path,
    *,
    experiments: set[str] | None,
    seeds: set[int] | None,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_experiment: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(runs_root.rglob(REEVAL_FILENAME)):
        payload = read_json(path)
        if not isinstance(payload, dict) or payload.get('status') != 'completed':
            continue
        identity = payload.get('run_identity') or {}
        experiment = identity.get('experiment')
        seed = identity.get('seed')
        if experiment is None or seed is None:
            raise ValueError(f'{path} is missing run_identity.experiment or run_identity.seed')
        experiment = str(experiment)
        seed = int(seed)
        if experiments is not None and experiment not in experiments:
            continue
        if seeds is not None and seed not in seeds:
            continue
        rows_by_experiment.setdefault(experiment, []).append(payload)
    if not rows_by_experiment:
        raise ValueError(f'no {REEVAL_FILENAME} files matched under {runs_root}')
    for experiment, rows in rows_by_experiment.items():
        rows.sort(key=lambda row: int(row['run_identity']['seed']))
        seen = set()
        for row in rows:
            seed = int(row['run_identity']['seed'])
            if seed in seen:
                raise ValueError(f'duplicate reevaluation for {experiment} seed {seed}')
            seen.add(seed)
    return rows_by_experiment


def existing_record_by_seed(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = payload.get('runs') if isinstance(payload.get('runs'), list) else []
    return {
        int(record['seed']): record
        for record in records
        if isinstance(record, dict) and record.get('seed') is not None
    }


def record_from_reevaluation(row: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = existing or {}
    identity = row['run_identity']
    test_metrics = row['metrics']['test']
    metrics_05 = test_metrics['threshold_0_5']
    metrics_exact = test_metrics['exact_val_best']
    record: dict[str, Any] = {
        'seed': int(identity['seed']),
        'status': 'completed',
        'run_dir': identity.get('run_dir'),
        'split_json': row.get('split_path'),
        'best_epoch': existing.get('best_epoch'),
        'best_val_threshold': float(row['exact_validation_optimal_threshold']),
        'resolution_tag': row.get('split', {}).get('resolution_tag') or existing.get('resolution_tag'),
        'filtered_graph_count': existing.get('filtered_graph_count'),
    }
    record.update(metric_columns(THRESHOLD_05_PREFIX, metrics_05))
    record.update(metric_columns(VAL_BEST_PREFIX, metrics_exact))
    if 'error' in existing:
        record['error'] = None
    return record


def update_seed_summary(row: dict[str, Any]) -> None:
    run_dir = Path(row['run_identity']['run_dir'])
    summary_path = run_dir / 'summary.json'
    if not summary_path.exists():
        raise ValueError(f'missing seed summary: {summary_path}')
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f'{summary_path} must contain an object')
    exact_metrics = row['metrics']['test']['exact_val_best']
    half_metrics = row['metrics']['test']['threshold_0_5']
    summary['best_validation_threshold'] = float(row['exact_validation_optimal_threshold'])
    summary['test_metrics_threshold_0_5'] = half_metrics
    summary['test_metrics_best_validation_threshold'] = exact_metrics
    summary['test_confusion_threshold_0_5'] = confusion_counts(half_metrics)
    summary['test_confusion_best_validation_threshold'] = confusion_counts(exact_metrics)
    write_json(summary_path, summary)


def materialize_reeval_reports(args: Namespace) -> None:
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise ValueError(f'--runs-root does not exist: {runs_root}')
    requested_experiments = set(args.experiments) if args.experiments else None
    requested_seeds = {int(seed) for seed in args.seeds} if args.seeds else None
    rows_by_experiment = load_reevaluation_rows(
        runs_root,
        experiments=requested_experiments,
        seeds=requested_seeds,
    )
    if getattr(args, 'dry_run', False):
        for experiment, rows in rows_by_experiment.items():
            seeds = ', '.join(str(row['run_identity']['seed']) for row in rows)
            print(f'would materialize: experiment={experiment} seeds={seeds}')
        return

    payloads = load_existing_suite_payloads(runs_root)
    for experiment, rows in rows_by_experiment.items():
        experiment_dir = runs_root / 'experiments' / experiment
        payload = dict(payloads.get(experiment) or {})
        old_records = existing_record_by_seed(payload)
        if requested_seeds is None and old_records:
            row_seeds = {int(row['run_identity']['seed']) for row in rows}
            required_seeds = {
                seed
                for seed, record in old_records.items()
                if record.get('status') == 'completed'
            }
            missing = sorted(required_seeds - row_seeds)
            if missing:
                raise ValueError(
                    f'{experiment} has only partial reevaluation results; missing seed(s): {missing}. '
                    f'Run full reevaluation first, or pass --seeds for an intentional partial materialization.'
                )
        records = [
            record_from_reevaluation(row, old_records.get(int(row['run_identity']['seed'])))
            for row in rows
        ]
        payload['experiment'] = payload.get('experiment') or experiment
        payload['runs'] = records
        payload['aggregates'] = aggregate_records(records)
        write_experiment_reports(experiment_dir, payload)
        payloads[experiment] = payload
        if getattr(args, 'update_run_summaries', False):
            for row in rows:
                update_seed_summary(row)

    write_suite_reports(runs_root, payloads)
    print(f'materialized exact-threshold reports -> {runs_root}')
