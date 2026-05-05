import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


METRIC_KEYS = ('f1', 'precision', 'recall', 'fpr', 'tpr', 'accuracy')
DISPLAY_METRIC_KEYS = ('f1', 'precision', 'recall', 'fpr', 'accuracy')
DISPLAY_METRIC_LABELS = {'recall': 'rec(tpr)'}
THRESHOLD_05_PREFIX = 'test_0_5'
VAL_BEST_PREFIX = 'test_val_best'


def build_train_command(
    *,
    dataset: str,
    run_dir: Path,
    split_json: Path,
    seed: int,
    resolution_tag: str,
    epochs: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path('tools') / 'run_baseline.py'),
        '--model',
        'graphsage',
        '--dataset',
        dataset,
        '--run-dir',
        str(run_dir),
        '--preset',
        'paper',
        '--resolution-tag',
        resolution_tag,
        '--seed',
        str(seed),
        '--split-json-out',
        str(split_json),
        '--epochs',
        str(epochs),
    ]


def _metric_columns(prefix: str) -> list[str]:
    return [f'{prefix}_{metric}' for metric in METRIC_KEYS]


def _display_metric_label(metric: str) -> str:
    return DISPLAY_METRIC_LABELS.get(metric, metric)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def collect_success_record(seed: int, run_dir: Path, split_json: Path) -> dict[str, Any]:
    summary_path = run_dir / 'summary.json'
    summary = _read_json(summary_path)
    metrics_05 = summary.get('test_metrics_threshold_0_5', {})
    metrics_best = summary.get('test_metrics_best_validation_threshold', {})

    record: dict[str, Any] = {
        'seed': seed,
        'status': 'completed',
        'run_dir': str(run_dir),
        'split_json': str(split_json),
        'best_epoch': summary.get('best_epoch'),
        'best_val_threshold': summary.get('best_validation_threshold'),
        'resolution_selector': summary.get('resolution_selector', summary.get('resolution_tag')),
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
        'resolution_selector': None,
        'filtered_graph_count': None,
        'error': error,
    }
    for column in _metric_columns(THRESHOLD_05_PREFIX) + _metric_columns(VAL_BEST_PREFIX):
        record[column] = None
    return record


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for prefix in (THRESHOLD_05_PREFIX, VAL_BEST_PREFIX):
        aggregates[prefix] = {}
        for metric in METRIC_KEYS:
            column = f'{prefix}_{metric}'
            values = [
                float(record[column])
                for record in records
                if record.get('status') == 'completed' and record.get(column) is not None
            ]
            if values:
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                aggregates[prefix][metric] = {
                    'mean': statistics.mean(values),
                    'std': std,
                    'n': len(values),
                }
            else:
                aggregates[prefix][metric] = {'mean': None, 'std': None, 'n': 0}
    return aggregates


def build_summary_payload(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'experiment': 'graphseam_paper_baseline',
        'dataset': args.dataset,
        'resolution_tag': args.resolution_tag,
        'resolution_selector': args.resolution_tag,
        'epochs': args.epochs,
        'seeds': args.seeds,
        'runs': records,
        'aggregates': aggregate_records(records),
    }


def write_reports(output_root: Path, payload: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    records = payload['runs']
    columns = [
        'seed',
        'status',
        'run_dir',
        'split_json',
        'best_epoch',
        'best_val_threshold',
        'resolution_selector',
        'filtered_graph_count',
        *_metric_columns(THRESHOLD_05_PREFIX),
        *_metric_columns(VAL_BEST_PREFIX),
        'error',
    ]

    with open(output_root / 'baseline_summary.json', 'w') as f:
        json.dump(payload, f, indent=2)

    with open(output_root / 'baseline_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['row_type', *columns])
        writer.writeheader()
        for record in records:
            writer.writerow({'row_type': 'seed', **record})
        for row_type in ('mean', 'std'):
            row: dict[str, Any] = {'row_type': row_type}
            for prefix in (THRESHOLD_05_PREFIX, VAL_BEST_PREFIX):
                for metric in METRIC_KEYS:
                    row[f'{prefix}_{metric}'] = payload['aggregates'][prefix][metric][row_type]
            writer.writerow(row)

    with open(output_root / 'baseline_summary.md', 'w') as f:
        f.write(_markdown_summary(payload))


def _fmt(value: Any) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        '# GraphSeam Paper Baseline',
        '',
        '| seed | status | epoch | val threshold | F1 @0.5 | F1 @val-best |',
        '|---:|---|---:|---:|---:|---:|',
    ]
    for record in payload['runs']:
        lines.append(
            '| '
            f"{record['seed']} | {record['status']} | {_fmt(record.get('best_epoch'))} | "
            f"{_fmt(record.get('best_val_threshold'))} | "
            f"{_fmt(record.get(f'{THRESHOLD_05_PREFIX}_f1'))} | "
            f"{_fmt(record.get(f'{VAL_BEST_PREFIX}_f1'))} |"
        )

    aggregates = payload['aggregates']
    lines.extend([
        '',
        '| metric | mean @0.5 | std @0.5 | mean @val-best | std @val-best |',
        '|---|---:|---:|---:|---:|',
    ])
    for metric in DISPLAY_METRIC_KEYS:
        lines.append(
            '| '
            f'{_display_metric_label(metric)} | '
            f"{_fmt(aggregates[THRESHOLD_05_PREFIX][metric]['mean'])} | "
            f"{_fmt(aggregates[THRESHOLD_05_PREFIX][metric]['std'])} | "
            f"{_fmt(aggregates[VAL_BEST_PREFIX][metric]['mean'])} | "
            f"{_fmt(aggregates[VAL_BEST_PREFIX][metric]['std'])} |"
        )
    lines.append('')
    return '\n'.join(lines)


def print_seed_table(record: dict[str, Any]) -> None:
    print('seed  status      epoch  val_t   f1@0.5  f1@val-best')
    print('----  ----------  -----  -----  ------  -----------')
    print(
        f"{record['seed']:>4}  {record['status']:<10}  {_fmt(record.get('best_epoch')):>5}  "
        f"{_fmt(record.get('best_val_threshold')):>5}  "
        f"{_fmt(record.get(f'{THRESHOLD_05_PREFIX}_f1')):>6}  "
        f"{_fmt(record.get(f'{VAL_BEST_PREFIX}_f1')):>11}"
    )


def print_aggregate_table(payload: dict[str, Any]) -> None:
    print('\naggregate over completed runs')
    print('metric     mean@0.5  std@0.5  mean@val-best  std@val-best')
    print('--------  --------  -------  -------------  ------------')
    aggregates = payload['aggregates']
    for metric in DISPLAY_METRIC_KEYS:
        print(
            f'{_display_metric_label(metric):<8}  '
            f"{_fmt(aggregates[THRESHOLD_05_PREFIX][metric]['mean']):>8}  "
            f"{_fmt(aggregates[THRESHOLD_05_PREFIX][metric]['std']):>7}  "
            f"{_fmt(aggregates[VAL_BEST_PREFIX][metric]['mean']):>13}  "
            f"{_fmt(aggregates[VAL_BEST_PREFIX][metric]['std']):>12}"
        )


def run_batch(args: argparse.Namespace, runner=subprocess.run) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    records: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_dir = output_root / f'seed_{seed}'
        split_json = output_root / 'splits' / f'seed_{seed}.json'
        run_dir.mkdir(parents=True, exist_ok=True)
        split_json.parent.mkdir(parents=True, exist_ok=True)

        command = build_train_command(
            dataset=args.dataset,
            run_dir=run_dir,
            split_json=split_json,
            seed=seed,
            resolution_tag=args.resolution_tag,
            epochs=args.epochs,
        )

        print(f'\nseed {seed}: running baseline')
        try:
            runner(command, check=True)
            record = collect_success_record(seed, run_dir, split_json)
        except subprocess.CalledProcessError as exc:
            record = failure_record(seed, run_dir, split_json, f'baseline runner exited with {exc.returncode}')
            records.append(record)
            print_seed_table(record)
            if not args.keep_going:
                return records
            continue
        except Exception as exc:
            record = failure_record(seed, run_dir, split_json, str(exc))
            records.append(record)
            print_seed_table(record)
            if not args.keep_going:
                return records
            continue

        records.append(record)
        print_seed_table(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run multi-seed GraphSeam baseline batches via tools/run_baseline.py.'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--resolution-tag', default='all')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument(
        '--keep-going',
        action='store_true',
        help='continue after failed seed runs',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    records = run_batch(args)
    payload = build_summary_payload(args, records)
    write_reports(output_root, payload)
    print_aggregate_table(payload)
    print(f'\nreports saved -> {output_root}')

    if any(record['status'] == 'failed' for record in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
