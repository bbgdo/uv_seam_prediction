import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tools.evaluate_saved_models import (
    aggregate_reevaluations,
    build_report_grid,
    compute_threshold_metrics_fast,
    discover_saved_runs,
    exact_validation_threshold,
    load_reference_control_reevaluations,
)
from tools.utils.reeval_runs import feature_selection_from_config, load_state_dict
from tools.utils.reeval_thresholds import best_threshold_index


def _metrics(
    f1: float,
    fpr: float,
    *,
    precision: float = 0.7,
    recall: float = 0.6,
    accuracy: float = 0.8,
) -> dict:
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tpr': recall,
        'accuracy': accuracy,
        'threshold': 0.9,
    }


def _reeval(
    experiment: str,
    seed: int,
    exact_f1: float,
    half_f1: float,
    delta_f1: float,
    *,
    split_path: Path | None = None,
    precision: float = 0.7,
    recall: float = 0.6,
    fpr: float = 0.05,
    accuracy: float = 0.8,
) -> dict:
    return {
        'status': 'completed',
        'run_identity': {
            'experiment': experiment,
            'seed': seed,
            'run_dir': f'runs/{experiment}/seed_{seed}',
        },
        'split_path': str(split_path) if split_path is not None else f'splits/seed_{seed}.json',
        'metrics': {
            'test': {
                'exact_val_best': _metrics(
                    exact_f1,
                    fpr,
                    precision=precision,
                    recall=recall,
                    accuracy=accuracy,
                ),
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


def _write_split(path: Path, *, seed: int, train: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'train_group_ids': train or ['train'],
        'val_group_ids': ['val'],
        'test_group_ids': ['test'],
        'seed': seed,
        'group_mode': 'family',
        'dataset_path': None,
        'resolution_tag': 'all',
    }))


def _write_reference_reeval(
    reference_dir: Path,
    *,
    seed: int,
    split_path: Path,
    f1: float,
    precision: float = 0.7,
) -> None:
    run_dir = reference_dir / f'seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = _reeval(
        'control14',
        seed,
        exact_f1=f1,
        half_f1=0.1,
        delta_f1=0.0,
        split_path=split_path,
        precision=precision,
    )
    (run_dir / 'reeval_exact_threshold.json').write_text(json.dumps(payload))


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
    def test_feature_selection_from_config_includes_thickness_sdf(self):
        selection = feature_selection_from_config({
            'feature_group': 'custom',
            'feature_flags': {
                'thickness_sdf': True,
            },
        })

        self.assertEqual(selection.feature_group, 'custom')
        self.assertTrue(selection.feature_flags.thickness_sdf)
        self.assertIn('thickness_sdf', selection.feature_names)

    def test_load_state_dict_rejects_wrapper_checkpoints(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best_model.pth'
            torch.save({'weight': torch.zeros(1)}, path)
            self.assertIn('weight', load_state_dict(path, torch.device('cpu')))

            torch.save({'state_dict': {'weight': torch.zeros(1)}}, path)
            with self.assertRaisesRegex(ValueError, 'state dict'):
                load_state_dict(path, torch.device('cpu'))

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
        best_index = best_threshold_index(
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

    def test_discover_saved_run_uses_gnn_dataset_override(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / 'experiments' / 'control14' / 'seed_7'
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
                'skip_connections': 'all',
            }))
            (run_dir / 'summary.json').write_text(json.dumps({'best_validation_threshold': 0.95}))
            splits_dir = root / 'splits'
            splits_dir.mkdir()
            (splits_dir / 'seed_7.json').write_text('{}')

            targets = discover_saved_runs(Namespace(
                runs_root=str(root),
                splits_dir=str(splits_dir),
                gnn_dataset='custom_override.pt',
                experiments=['control14'],
                seeds=[7],
            ))

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].dataset_path, Path('custom_override.pt'))
        self.assertEqual(targets[0].experiment, 'control14')

    def test_aggregate_json_structure_from_mocked_reevaluations(self):
        payload = aggregate_reevaluations([
            _reeval('control14', 1, exact_f1=0.50, half_f1=0.40, delta_f1=0.02),
            _reeval('control14', 2, exact_f1=0.70, half_f1=0.60, delta_f1=0.03),
            _reeval('ao_only', 1, exact_f1=0.55, half_f1=0.45, delta_f1=0.04),
            _reeval('ao_only', 2, exact_f1=0.72, half_f1=0.62, delta_f1=0.05),
        ])

        self.assertEqual(payload['run_count'], 4)
        self.assertIn('control14', payload['experiments'])
        self.assertAlmostEqual(
            payload['experiments']['control14']['test_exact_threshold']['f1']['mean'],
            0.60,
        )
        self.assertAlmostEqual(
            payload['experiments']['ao_only']['paired_delta_vs_old_stored_val_best']['f1']['mean'],
            0.045,
        )
        self.assertEqual(
            payload['paired_delta_vs_control14']['ao_only']['paired_seed_count'],
            2,
        )
        self.assertNotIn('external_reference_control', payload)

    def test_load_reference_control_reevaluations_by_seed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_path = root / 'splits' / 'seed_1.json'
            _write_split(split_path, seed=1)
            reference_dir = root / 'experiments' / 'control14'
            _write_reference_reeval(reference_dir, seed=1, split_path=split_path, f1=0.42)

            reference = load_reference_control_reevaluations(reference_dir)

        self.assertEqual(reference.experiment_name, 'control14')
        self.assertEqual(sorted(reference.by_seed), [1])
        self.assertAlmostEqual(
            reference.by_seed[1]['payload']['metrics']['test']['exact_val_best']['f1'],
            0.42,
        )
        self.assertIsNotNone(reference.by_seed[1]['split_identity']['fingerprint'])

    def test_external_control_pairing_succeeds_with_matching_seed_and_split(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_path = root / 'splits' / 'seed_1.json'
            _write_split(split_path, seed=1)
            reference_dir = root / 'experiments' / 'control14'
            _write_reference_reeval(
                reference_dir,
                seed=1,
                split_path=split_path,
                f1=0.50,
                precision=0.60,
            )
            reference = load_reference_control_reevaluations(reference_dir)

            payload = aggregate_reevaluations(
                [
                    _reeval(
                        'ao_density',
                        1,
                        exact_f1=0.62,
                        half_f1=0.30,
                        delta_f1=0.0,
                        split_path=split_path,
                        precision=0.65,
                    ),
                ],
                reference_control=reference,
            )

        delta = payload['paired_delta_vs_reference_control']['ao_density']
        self.assertEqual(delta['paired_seed_count'], 1)
        self.assertEqual(delta['paired_seeds'], [1])
        self.assertAlmostEqual(delta['summary']['f1']['mean'], 0.12)
        self.assertAlmostEqual(delta['summary']['precision']['mean'], 0.05)
        self.assertEqual(delta['per_seed'][0]['identity_check'], 'split_content_fingerprint')

    def test_external_control_pairing_records_split_mismatch_skip(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_1 = root / 'splits' / 'seed_1.json'
            split_2_control = root / 'splits' / 'control_seed_2.json'
            split_2_target = root / 'splits' / 'target_seed_2.json'
            _write_split(split_1, seed=1)
            _write_split(split_2_control, seed=2, train=['control_train'])
            _write_split(split_2_target, seed=2, train=['target_train'])
            reference_dir = root / 'experiments' / 'control14'
            _write_reference_reeval(reference_dir, seed=1, split_path=split_1, f1=0.50)
            _write_reference_reeval(reference_dir, seed=2, split_path=split_2_control, f1=0.60)
            reference = load_reference_control_reevaluations(reference_dir)

            payload = aggregate_reevaluations(
                [
                    _reeval('ao_symmetry', 1, 0.55, 0.30, 0.0, split_path=split_1),
                    _reeval('ao_symmetry', 2, 0.70, 0.30, 0.0, split_path=split_2_target),
                ],
                reference_control=reference,
            )

        delta = payload['paired_delta_vs_reference_control']['ao_symmetry']
        self.assertEqual(delta['paired_seed_count'], 1)
        self.assertEqual(delta['skipped_seeds'][0]['seed'], 2)
        self.assertEqual(delta['skipped_seeds'][0]['reason'], 'split_content_fingerprint_mismatch')

    def test_external_control_pairing_fails_when_all_splits_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            control_split = root / 'splits' / 'control_seed_1.json'
            target_split = root / 'splits' / 'target_seed_1.json'
            _write_split(control_split, seed=1, train=['control_train'])
            _write_split(target_split, seed=1, train=['target_train'])
            reference_dir = root / 'experiments' / 'control14'
            _write_reference_reeval(reference_dir, seed=1, split_path=control_split, f1=0.50)
            reference = load_reference_control_reevaluations(reference_dir)

            with self.assertRaisesRegex(ValueError, 'no valid external control pairings'):
                aggregate_reevaluations(
                    [_reeval('ao_density', 1, 0.55, 0.30, 0.0, split_path=target_split)],
                    reference_control=reference,
                )

    def test_external_control_deltas_use_per_seed_metrics_not_control_aggregate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_dir = root / 'experiments' / 'control14'
            target_rows = []
            for seed, control_f1, target_f1 in [
                (1, 0.10, 0.50),
                (2, 0.70, 0.80),
                (3, 0.99, None),
            ]:
                split_path = root / 'splits' / f'seed_{seed}.json'
                _write_split(split_path, seed=seed)
                _write_reference_reeval(reference_dir, seed=seed, split_path=split_path, f1=control_f1)
                if target_f1 is not None:
                    target_rows.append(
                        _reeval('ao_density', seed, target_f1, 0.30, 0.0, split_path=split_path)
                    )
            reference = load_reference_control_reevaluations(reference_dir)

            payload = aggregate_reevaluations(target_rows, reference_control=reference)

        delta = payload['paired_delta_vs_reference_control']['ao_density']
        self.assertEqual(delta['paired_seed_count'], 2)
        self.assertAlmostEqual(delta['per_seed'][0]['f1'], 0.40)
        self.assertAlmostEqual(delta['per_seed'][1]['f1'], 0.10)
        self.assertAlmostEqual(delta['summary']['f1']['mean'], 0.25)


if __name__ == '__main__':
    unittest.main()
