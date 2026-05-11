from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .ablation_datasets import get_gnn_dataset_arg
from .ablation_specs import (
    BASELINE_EXPERIMENT,
    ExperimentSpec,
    experiment_feature_selection,
    is_meshcnn_model,
)


METRIC_KEYS = ('f1', 'precision', 'recall', 'fpr', 'tpr', 'accuracy')
DELTA_METRIC_KEYS = ('fpr', 'recall', 'f1', 'accuracy')
THRESHOLD_05_PREFIX = 'test_0_5'
VAL_BEST_PREFIX = 'test_val_best'


def _metric_columns(prefix: str) -> list[str]:
    return [f'{prefix}_{metric}' for metric in METRIC_KEYS]


def _delta_columns(prefix: str) -> list[str]:
    return [f'delta_{prefix}_{metric}' for metric in DELTA_METRIC_KEYS]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def collect_success_record(seed: int, run_dir: Path, split_json: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / 'summary.json')
    metrics_05 = summary.get('test_metrics_threshold_0_5', {})
    metrics_best = summary.get('test_metrics_best_validation_threshold', {})

    record: dict[str, Any] = {
        'seed': seed,
        'status': 'completed',
        'run_dir': str(run_dir),
        'split_json': str(split_json),
        'best_epoch': summary.get('best_epoch'),
        'best_val_threshold': summary.get('best_validation_threshold'),
        'resolution_tag': summary.get('resolution_tag'),
        'filtered_graph_count': summary.get('filtered_graph_count'),
    }
    for metric in METRIC_KEYS:
        record[f'{THRESHOLD_05_PREFIX}_{metric}'] = metrics_05.get(metric)
        record[f'{VAL_BEST_PREFIX}_{metric}'] = metrics_best.get(metric)
    return record


def failure_record(seed: int, run_dir: Path, split_json: Path, error: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        'seed': seed,
        'status': 'failed',
        'run_dir': str(run_dir),
        'split_json': str(split_json),
        'best_epoch': None,
        'best_val_threshold': None,
        'resolution_tag': None,
        'filtered_graph_count': None,
        'error': error,
    }
    for column in _metric_columns(THRESHOLD_05_PREFIX) + _metric_columns(VAL_BEST_PREFIX):
        record[column] = None
    return record


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    aggregates: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for prefix in (THRESHOLD_05_PREFIX, VAL_BEST_PREFIX):
        aggregates[prefix] = {}
        for metric in METRIC_KEYS:
            column = f'{prefix}_{metric}'
            values = [
                float(record[column])
                for record in records
                if record.get('status') == 'completed' and record.get(column) is not None
            ]
            aggregates[prefix][metric] = {
                'mean': statistics.mean(values) if values else None,
                'std': statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
                'n': len(values),
            }
    return aggregates


def paired_delta_summary(
    *,
    experiment_name: str,
    experiment_records: list[dict[str, Any]],
    control_name: str,
    control_records: list[dict[str, Any]],
) -> dict[str, Any]:
    experiment_by_seed = {
        int(record['seed']): record for record in experiment_records if record.get('status') == 'completed'
    }
    control_by_seed = {
        int(record['seed']): record for record in control_records if record.get('status') == 'completed'
    }
    paired_seeds = sorted(set(experiment_by_seed) & set(control_by_seed))

    rows: list[dict[str, Any]] = []
    for seed in paired_seeds:
        row: dict[str, Any] = {'seed': seed}
        experiment_record = experiment_by_seed[seed]
        control_record = control_by_seed[seed]
        for prefix in (VAL_BEST_PREFIX, THRESHOLD_05_PREFIX):
            for metric in DELTA_METRIC_KEYS:
                column = f'{prefix}_{metric}'
                row[f'delta_{column}'] = (
                    float(experiment_record[column]) - float(control_record[column])
                    if experiment_record.get(column) is not None and control_record.get(column) is not None
                    else None
                )
        rows.append(row)

    def summarize(prefix: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for metric in DELTA_METRIC_KEYS:
            column = f'delta_{prefix}_{metric}'
            values = [float(row[column]) for row in rows if row.get(column) is not None]
            payload[column] = {
                'mean': statistics.mean(values) if values else None,
                'std': statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
                'n': len(values),
            }
        payload['win_count_fpr'] = sum(
            1 for row in rows if row.get(f'delta_{prefix}_fpr') is not None and row[f'delta_{prefix}_fpr'] < 0
        )
        payload['win_count_f1'] = sum(
            1 for row in rows if row.get(f'delta_{prefix}_f1') is not None and row[f'delta_{prefix}_f1'] > 0
        )
        return payload

    return {
        'experiment': experiment_name,
        'control': control_name,
        'paired_seed_count': len(rows),
        'paired_seeds': paired_seeds,
        'per_seed': rows,
        'val_best': summarize(VAL_BEST_PREFIX),
        'threshold_0_5_diagnostics': summarize(THRESHOLD_05_PREFIX),
    }


def build_experiment_payload(
    *,
    args: argparse.Namespace,
    spec: ExperimentSpec,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    selection = experiment_feature_selection(spec.name)
    model = getattr(args, 'model', 'graphsage')
    dataset = args.meshcnn_dataset if is_meshcnn_model(model) else get_gnn_dataset_arg(args)
    return {
        'experiment': spec.name,
        'phase': spec.phase,
        'model': model,
        'dataset': dataset,
        'feature_group': spec.feature_group,
        'feature_flags': selection.feature_flags.as_dict(),
        'feature_names': list(selection.feature_names),
        'resolution_tag': args.resolution_tag,
        'epochs': args.epochs,
        'patience': getattr(args, 'patience', 15),
        'seeds': args.seeds,
        'splits_dir': str(args.splits_dir),
        'split_json_in': str(args.split_json_in) if getattr(args, 'split_json_in', None) else None,
        'runs': records,
        'aggregates': aggregate_records(records),
    }


def write_experiment_reports(experiment_dir: Path, payload: dict[str, Any]) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_dir / 'summary.json', payload)
    _write_json(experiment_dir / 'per_seed_summary.json', {'runs': payload['runs']})
    _write_json(experiment_dir / 'aggregate_summary.json', payload['aggregates'])

    columns = [
        'seed',
        'status',
        'run_dir',
        'split_json',
        'best_epoch',
        'best_val_threshold',
        'resolution_tag',
        'filtered_graph_count',
        *_metric_columns(THRESHOLD_05_PREFIX),
        *_metric_columns(VAL_BEST_PREFIX),
        'error',
    ]
    with (experiment_dir / 'per_seed_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in payload['runs']:
            writer.writerow(record)

    with (experiment_dir / 'aggregate_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['prefix', 'metric', 'mean', 'std', 'n'])
        writer.writeheader()
        for prefix in (VAL_BEST_PREFIX, THRESHOLD_05_PREFIX):
            for metric in METRIC_KEYS:
                row = payload['aggregates'][prefix][metric]
                writer.writerow({'prefix': prefix, 'metric': metric, **row})


def write_delta_reports(path_prefix: Path, payload: dict[str, Any]) -> None:
    _write_json(path_prefix.with_suffix('.json'), payload)
    columns = ['seed', *_delta_columns(VAL_BEST_PREFIX), *_delta_columns(THRESHOLD_05_PREFIX)]
    with path_prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in payload['per_seed']:
            writer.writerow(row)


def print_experiment_result(name: str, records: list[dict[str, Any]]) -> None:
    completed = [record for record in records if record.get('status') == 'completed']
    failed = len(records) - len(completed)
    if not completed:
        print(f"{name}: 0/{len(records)} completed")
        return
    f1_values = [float(record[f'{VAL_BEST_PREFIX}_f1']) for record in completed]
    fpr_values = [float(record[f'{VAL_BEST_PREFIX}_fpr']) for record in completed]
    print(
        f"{name}: {len(completed)}/{len(records)} completed, "
        f"F1 {statistics.mean(f1_values):.4f}, FPR {statistics.mean(fpr_values):.4f}"
        + (f", failed {failed}" if failed else '')
    )


def load_existing_suite_payloads(output_root: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    suite_path = output_root / 'suite_summary.json'
    if suite_path.exists():
        suite_payload = _read_json(suite_path)
        existing = suite_payload.get('experiments', {})
        if isinstance(existing, dict):
            payloads.update(existing)

    experiments_dir = output_root / 'experiments'
    if experiments_dir.exists():
        for summary_path in experiments_dir.glob('*/summary.json'):
            payload = _read_json(summary_path)
            name = payload.get('experiment') or summary_path.parent.name
            payloads[str(name)] = payload
    return payloads


def write_suite_reports(output_root: Path, payloads: dict[str, dict[str, Any]]) -> None:
    _write_json(
        output_root / 'suite_summary.json',
        {
            'strategy': 'pairwise_feature_search',
            'experiments_per_architecture': len(payloads),
            'experiments': payloads,
        },
    )

    if BASELINE_EXPERIMENT in payloads:
        control_records = payloads[BASELINE_EXPERIMENT]['runs']
        suite_deltas: dict[str, Any] = {}
        for name, payload in payloads.items():
            delta = paired_delta_summary(
                experiment_name=name,
                experiment_records=payload['runs'],
                control_name=BASELINE_EXPERIMENT,
                control_records=control_records,
            )
            suite_deltas[name] = delta
            write_delta_reports(
                output_root / 'experiments' / name / 'paired_delta_vs_control14',
                delta,
            )
        _write_json(output_root / 'paired_deltas_vs_control14.json', suite_deltas)
