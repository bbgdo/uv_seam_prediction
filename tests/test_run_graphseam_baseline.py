import csv
import json
import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.run_graphseam_baseline import (
    THRESHOLD_05_PREFIX,
    VAL_BEST_PREFIX,
    aggregate_records,
    build_summary_payload,
    build_train_command,
    parse_args,
    run_batch,
    write_reports,
)


def _summary(seed: int, f1_05: float, f1_best: float) -> dict:
    metrics_05 = {
        'f1': f1_05,
        'precision': 0.5 + seed / 100,
        'recall': 0.6,
        'fpr': 0.1,
        'tpr': 0.6,
        'accuracy': 0.7,
    }
    metrics_best = {
        'f1': f1_best,
        'precision': 0.55 + seed / 100,
        'recall': 0.65,
        'fpr': 0.08,
        'tpr': 0.65,
        'accuracy': 0.75,
    }
    return {
        'best_epoch': seed + 1,
        'best_validation_threshold': 0.6,
        'resolution_selector': 'all',
        'filtered_graph_count': 12,
        'test_metrics_threshold_0_5': metrics_05,
        'test_metrics_best_validation_threshold': metrics_best,
    }


def _args(tmp: Path, seeds: list[int], keep_going: bool = False) -> Namespace:
    return Namespace(
        dataset='dataset_paper14_dual.pt',
        output_root=str(tmp),
        seeds=seeds,
        resolution_tag='all',
        group_mode='family',
        preset='paper',
        epochs=3,
        keep_going=keep_going,
    )


class GraphSeamBaselineRunnerTests(unittest.TestCase):
    def test_summary_aggregation_from_mocked_run_summaries(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for seed, f1_05, f1_best in [(1, 0.4, 0.5), (2, 0.6, 0.7)]:
                run_dir = root / f'seed_{seed}'
                run_dir.mkdir()
                with open(run_dir / 'summary.json', 'w') as f:
                    json.dump(_summary(seed, f1_05, f1_best), f)
                records.append({
                    'seed': seed,
                    'status': 'completed',
                    'run_dir': str(run_dir),
                    'split_json': str(root / 'splits' / f'seed_{seed}.json'),
                    'best_epoch': seed + 1,
                    'best_val_threshold': 0.6,
                    f'{THRESHOLD_05_PREFIX}_f1': f1_05,
                    f'{THRESHOLD_05_PREFIX}_precision': 0.5 + seed / 100,
                    f'{THRESHOLD_05_PREFIX}_recall': 0.6,
                    f'{THRESHOLD_05_PREFIX}_fpr': 0.1,
                    f'{THRESHOLD_05_PREFIX}_tpr': 0.6,
                    f'{THRESHOLD_05_PREFIX}_accuracy': 0.7,
                    f'{VAL_BEST_PREFIX}_f1': f1_best,
                    f'{VAL_BEST_PREFIX}_precision': 0.55 + seed / 100,
                    f'{VAL_BEST_PREFIX}_recall': 0.65,
                    f'{VAL_BEST_PREFIX}_fpr': 0.08,
                    f'{VAL_BEST_PREFIX}_tpr': 0.65,
                    f'{VAL_BEST_PREFIX}_accuracy': 0.75,
                })

            aggregates = aggregate_records(records)

        self.assertAlmostEqual(aggregates[THRESHOLD_05_PREFIX]['f1']['mean'], 0.5)
        self.assertAlmostEqual(aggregates[VAL_BEST_PREFIX]['f1']['mean'], 0.6)
        self.assertAlmostEqual(aggregates[THRESHOLD_05_PREFIX]['f1']['std'], 0.1414213562)

    def test_report_generation_writes_json_csv_and_markdown(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{
                'seed': 1,
                'status': 'completed',
                'run_dir': str(root / 'seed_1'),
                'split_json': str(root / 'splits' / 'seed_1.json'),
                'best_epoch': 2,
                'best_val_threshold': 0.6,
                f'{THRESHOLD_05_PREFIX}_f1': 0.4,
                f'{THRESHOLD_05_PREFIX}_precision': 0.5,
                f'{THRESHOLD_05_PREFIX}_recall': 0.6,
                f'{THRESHOLD_05_PREFIX}_fpr': 0.1,
                f'{THRESHOLD_05_PREFIX}_tpr': 0.6,
                f'{THRESHOLD_05_PREFIX}_accuracy': 0.7,
                f'{VAL_BEST_PREFIX}_f1': 0.5,
                f'{VAL_BEST_PREFIX}_precision': 0.55,
                f'{VAL_BEST_PREFIX}_recall': 0.65,
                f'{VAL_BEST_PREFIX}_fpr': 0.08,
                f'{VAL_BEST_PREFIX}_tpr': 0.65,
                f'{VAL_BEST_PREFIX}_accuracy': 0.75,
            }]
            payload = build_summary_payload(_args(root, [1]), records)

            write_reports(root, payload)

            self.assertTrue((root / 'baseline_summary.json').exists())
            self.assertTrue((root / 'baseline_summary.csv').exists())
            self.assertTrue((root / 'baseline_summary.md').exists())
            with open(root / 'baseline_summary.csv', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row['row_type'] for row in rows], ['seed', 'mean', 'std'])
            self.assertIn('GraphSeam Paper Baseline', (root / 'baseline_summary.md').read_text())

    def test_subprocess_command_construction(self):
        command = build_train_command(
            dataset='dataset.pt',
            run_dir=Path('runs') / 'seed_7',
            split_json=Path('runs') / 'splits' / 'seed_7.json',
            seed=7,
            resolution_tag='10000f',
            group_mode='family',
            preset='paper',
            epochs=5,
        )

        self.assertEqual(command[0], sys.executable)
        self.assertIn(str(Path('tools') / 'run_baseline.py'), command)
        self.assertEqual(command[command.index('--model') + 1], 'graphsage')
        self.assertIn('--strict-paper-protocol', command)
        self.assertEqual(command[command.index('--resolution-tag') + 1], '10000f')
        self.assertEqual(command[command.index('--seed') + 1], '7')
        self.assertEqual(command[command.index('--split-json-out') + 1], str(Path('runs') / 'splits' / 'seed_7.json'))

    def test_parse_args_default_resolution_selector_is_all(self):
        argv = [
            'run_graphseam_baseline.py',
            '--dataset',
            'dataset.pt',
            '--output-root',
            'runs/baseline',
            '--seeds',
            '1',
            '--epochs',
            '1',
        ]

        with patch.object(sys, 'argv', argv):
            args = parse_args()

        self.assertEqual(args.resolution_tag, 'all')

    def test_keep_going_failure_handling_logic(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_runner(command, check):
                seed = int(command[command.index('--seed') + 1])
                run_dir = Path(command[command.index('--run-dir') + 1])
                if seed == 1:
                    raise subprocess.CalledProcessError(2, command)
                run_dir.mkdir(parents=True, exist_ok=True)
                with open(run_dir / 'summary.json', 'w') as f:
                    json.dump(_summary(seed, 0.6, 0.7), f)

            records = run_batch(_args(root, [1, 2], keep_going=True), runner=fake_runner)

        self.assertEqual([record['status'] for record in records], ['failed', 'completed'])
        self.assertIn('baseline runner exited with 2', records[0]['error'])

    def test_default_failure_handling_stops_batch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_runner(command, check):
                raise subprocess.CalledProcessError(2, command)

            records = run_batch(_args(root, [1, 2], keep_going=False), runner=fake_runner)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
