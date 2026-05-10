from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from tools.utils.json_io import read_json
from tools.utils.reeval_common import (
    DELTA_KEYS,
    METRIC_KEYS,
    PAIRED_DELTA_KEYS,
    REEVAL_FILENAME,
    ReferenceControlSet,
    exact_test_metrics,
    split_identities_match,
    split_identity,
)


def load_reference_control_reevaluations(reference_control_dir: str | Path) -> ReferenceControlSet:
    root = Path(reference_control_dir)
    if not root.exists():
        raise ValueError(f'--reference-control-dir does not exist: {root}')

    by_seed: dict[int, dict[str, Any]] = {}
    experiment_names = []
    for reeval_path in sorted(root.rglob(REEVAL_FILENAME)):
        payload = read_json(reeval_path)
        if not isinstance(payload, dict) or payload.get('status') != 'completed':
            continue
        if 'run_identity' not in payload or payload['run_identity'].get('seed') is None:
            raise ValueError(f'{reeval_path} is missing run_identity.seed')
        seed = int(payload['run_identity']['seed'])
        if seed in by_seed:
            raise ValueError(f'duplicate reference control reevaluation for seed {seed}: {reeval_path}')
        exact_test_metrics(payload, source=str(reeval_path))
        by_seed[seed] = {
            'payload': payload,
            'reeval_path': str(reeval_path),
            'split_identity': split_identity(payload, reeval_path=reeval_path),
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


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    return {
        'mean': statistics.mean(values) if values else None,
        'std': statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        'n': len(values),
    }


def summarize_metric_block(rows: list[dict[str, Any]], metric_path: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRIC_KEYS:
        values = []
        for row in rows:
            cursor: Any = row
            for key in metric_path:
                cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
            if isinstance(cursor, dict) and cursor.get(metric) is not None:
                values.append(float(cursor[metric]))
        summary[metric] = mean_std(values)
    return summary


def summarize_delta_block(rows: list[dict[str, Any]], delta_path: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in DELTA_KEYS:
        values = []
        for row in rows:
            cursor: Any = row
            for key in delta_path:
                cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
            if isinstance(cursor, dict) and cursor.get(metric) is not None:
                values.append(float(cursor[metric]))
        summary[metric] = mean_std(values)
    return summary


def paired_delta_vs_control(
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
        'summary': {metric: mean_std([float(row[metric]) for row in per_seed]) for metric in PAIRED_DELTA_KEYS},
    }


def paired_delta_vs_reference_control(
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

        target_identity = split_identity(target)
        control_identity = control['split_identity']
        matched, reason, method = split_identities_match(target_identity, control_identity)
        if not matched:
            skipped.append({
                'seed': seed,
                'reason': reason,
                'identity_check': method,
                'target_split_identity': target_identity,
                'control_split_identity': control_identity,
            })
            continue

        metrics = exact_test_metrics(target, source=f'{experiment} seed {seed}')
        control_metrics = exact_test_metrics(
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
        'summary': {metric: mean_std([float(row[metric]) for row in per_seed]) for metric in PAIRED_DELTA_KEYS},
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
            'test_exact_threshold': summarize_metric_block(rows, ('metrics', 'test', 'exact_val_best')),
            'test_threshold_0_5': summarize_metric_block(rows, ('metrics', 'test', 'threshold_0_5')),
            'paired_delta_vs_old_stored_val_best': summarize_delta_block(
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
            paired_vs_control[experiment] = paired_delta_vs_control(experiment, rows, control_rows)

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
            paired_vs_reference[experiment] = paired_delta_vs_reference_control(
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
