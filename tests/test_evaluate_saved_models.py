import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tools.evaluate_saved_models import (
    _best_threshold_index,
    aggregate_reevaluations,
    build_report_grid,
    compute_threshold_metrics_fast,
    discover_saved_runs,
    exact_validation_threshold,
)


def _metrics(f1: float, fpr: float) -> dict:
    return {
        'precision': 0.7,
        'recall': 0.6,
        'f1': f1,
        'fpr': fpr,
        'tpr': 0.6,
        'accuracy': 0.8,
        'threshold': 0.9,
    }


def _reeval(experiment: str, seed: int, exact_f1: float, half_f1: float, delta_f1: float) -> dict:
    return {
        'status': 'completed',
        'run_identity': {
            'experiment': experiment,
            'seed': seed,
            'run_dir': f'runs/{experiment}/seed_{seed}',
            'dataset_role': 'custom',
        },
        'metrics': {
            'test': {
                'exact_val_best': _metrics(exact_f1, 0.05),
                'threshold_0_5': _metrics(half_f1, 0.1),
            },
        },
        'comparison': {
            'delta_vs_old_stored_val_best': {
                'test': {
                    'fpr': -0.02,
                    'recall': 0.01,
                    'f1': delta_f1,
                    'accuracy': 0.03,
                },
            },
        },
    }


def _metrics_at_threshold_numpy(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = probs >= threshold
    gt = labels.astype(bool)
    tp = int(np.count_nonzero(preds & gt))
    fp = int(np.count_nonzero(preds & ~gt))
    fn = int(np.count_nonzero(~preds & gt))
    tn = int(np.count_nonzero(~preds & ~gt))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': (tp + tn) / max(len(gt), 1),
        'fpr': fp / max(fp + tn, 1),
        'tpr': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'threshold': float(threshold),
    }


def _bruteforce_exact_threshold(probs: np.ndarray, labels: np.ndarray) -> dict:
    candidates = {float(value) for value in probs if float(value) < 1.0}
    max_score = float(np.max(probs))
    above_max = float(np.nextafter(np.float64(max_score), np.float64(1.0)))
    if max_score < 1.0 and above_max < 1.0:
        candidates.add(above_max)
    elif not candidates:
        candidates.add(float(np.nextafter(np.float64(1.0), np.float64(0.0))))

    rows = [_metrics_at_threshold_numpy(probs, labels, threshold) for threshold in candidates]
    return max(
        rows,
        key=lambda row: (
            float(row['f1']),
            -float(row['fpr']),
            float(row['precision']),
            float(row['threshold']),
        ),
    )


class EvaluateSavedModelsTests(unittest.TestCase):
    def test_fast_threshold_matches_bruteforce_on_synthetic_example(self):
        probs = np.array([0.91, 0.83, 0.76, 0.65, 0.42, 0.21])
        labels = np.array([1, 0, 1, 1, 0, 0])

        result = compute_threshold_metrics_fast(probs, labels)
        expected = _bruteforce_exact_threshold(probs, labels)

        self.assertAlmostEqual(result['threshold'], expected['threshold'])
        self.assertAlmostEqual(result['metrics']['f1'], expected['f1'])
        self.assertEqual(result['metrics']['tp'], expected['tp'])
        self.assertEqual(result['metrics']['fp'], expected['fp'])

    def test_fast_threshold_collapses_repeated_probabilities(self):
        probs = np.array([0.7, 0.7, 0.3, 0.3])
        labels = np.array([1, 0, 1, 0])

        result = compute_threshold_metrics_fast(probs, labels)
        expected = _bruteforce_exact_threshold(probs, labels)

        self.assertEqual(result['candidate_count'], 3)
        self.assertAlmostEqual(result['threshold'], expected['threshold'])
        self.assertAlmostEqual(result['metrics']['f1'], expected['f1'])

    def test_threshold_tie_break_selector_order(self):
        best_index = _best_threshold_index(
            f1=np.array([0.7, 0.7, 0.7, 0.7]),
            fpr=np.array([0.2, 0.1, 0.1, 0.1]),
            precision=np.array([0.9, 0.8, 0.85, 0.85]),
            thresholds=np.array([0.9, 0.6, 0.7, 0.8]),
        )

        self.assertEqual(best_index, 3)

    def test_exact_threshold_excludes_threshold_one(self):
        probs = torch.tensor([1.0, 0.8, 0.4])
        labels = torch.tensor([1, 0, 0])

        result = exact_validation_threshold(probs, labels)

        self.assertNotEqual(result['threshold'], 1.0)
        self.assertEqual(result['candidate_count'], 2)

    def test_fast_threshold_exercises_vectorized_path_on_larger_array(self):
        rng = np.random.default_rng(123)
        probs = rng.random(50_000)
        labels = rng.integers(0, 2, size=50_000)

        result = compute_threshold_metrics_fast(probs, labels)

        self.assertGreater(result['candidate_count'], 1_000)
        self.assertIn('f1', result['metrics'])
        self.assertIn('threshold', result)

    def test_exact_threshold_can_select_above_old_grid_cap(self):
        probs = torch.tensor([0.96, 0.97, 0.98])
        labels = torch.tensor([0, 1, 1])

        result = exact_validation_threshold(probs, labels)

        self.assertAlmostEqual(result['threshold'], 0.97, places=6)
        self.assertAlmostEqual(result['metrics']['f1'], 1.0)

    def test_exact_threshold_tie_breaks_by_lower_fpr(self):
        probs = torch.tensor([0.2, 0.8])
        labels = torch.tensor([0, 0])

        result = exact_validation_threshold(probs, labels)

        self.assertGreater(result['threshold'], 0.8)
        self.assertEqual(result['metrics']['fpr'], 0.0)
        self.assertEqual(result['metrics']['f1'], 0.0)
        self.assertEqual(result['tie_breaking'][1], 'minimize validation fpr')

    def test_report_grid_excludes_threshold_one(self):
        self.assertNotIn(1.0, build_report_grid())
        custom = build_report_grid('0.99,1.0,0.995')
        self.assertEqual(custom, [0.99, 0.995])

    def test_default_dense_grid_generation(self):
        grid = build_report_grid()

        self.assertEqual(grid[:3], [0.9, 0.91, 0.92])
        self.assertEqual(grid[-2:], [0.995, 0.999])
        self.assertEqual(len(grid), 12)

    def test_discover_saved_run_selects_dataset_role_from_experiment(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / 'experiments' / 'custom14_control' / 'seed_7'
            run_dir.mkdir(parents=True)
            (run_dir / 'best_model.pth').write_bytes(b'checkpoint')
            (run_dir / 'config.json').write_text(json.dumps({
                'model_name': 'graphsage',
                'in_dim': 14,
                'hidden_dim': 64,
                'num_layers': 3,
                'dropout': 0.3,
                'dataset': 'original_custom.pt',
                'seed': 7,
                'group_mode': 'family',
                'feature_group': 'custom',
                'feature_flags': {},
                'resolution_tag': 'all',
                'aggr': 'lstm',
                'skip_connections': 'all',
            }))
            (run_dir / 'summary.json').write_text(json.dumps({'best_validation_threshold': 0.95}))
            splits_dir = root / 'splits'
            splits_dir.mkdir()
            (splits_dir / 'seed_7.json').write_text('{}')

            targets = discover_saved_runs(Namespace(
                runs_root=str(root),
                splits_dir=str(splits_dir),
                paper_dataset='paper_override.pt',
                custom_dataset='custom_override.pt',
                experiments=['custom14_control'],
                seeds=[7],
            ))

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].dataset_role, 'custom')
        self.assertEqual(targets[0].dataset_path, Path('custom_override.pt'))
        self.assertEqual(targets[0].experiment, 'custom14_control')

    def test_aggregate_json_structure_from_mocked_reevaluations(self):
        payload = aggregate_reevaluations([
            _reeval('custom14_control', 1, exact_f1=0.50, half_f1=0.40, delta_f1=0.02),
            _reeval('custom14_control', 2, exact_f1=0.70, half_f1=0.60, delta_f1=0.03),
            _reeval('ao_only', 1, exact_f1=0.55, half_f1=0.45, delta_f1=0.04),
            _reeval('ao_only', 2, exact_f1=0.72, half_f1=0.62, delta_f1=0.05),
        ])

        self.assertEqual(payload['run_count'], 4)
        self.assertIn('custom14_control', payload['experiments'])
        self.assertAlmostEqual(
            payload['experiments']['custom14_control']['test_exact_threshold']['f1']['mean'],
            0.60,
        )
        self.assertAlmostEqual(
            payload['experiments']['ao_only']['paired_delta_vs_old_stored_val_best']['f1']['mean'],
            0.045,
        )
        self.assertEqual(
            payload['paired_delta_vs_custom14_control']['ao_only']['paired_seed_count'],
            2,
        )


if __name__ == '__main__':
    unittest.main()
