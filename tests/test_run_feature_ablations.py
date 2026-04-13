import json
import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch_geometric.data import Data

from preprocessing.feature_registry import PAPER14_FEATURE_NAMES, resolve_feature_selection
from tools.run_feature_ablations import (
    EXPERIMENT_SPECS,
    THRESHOLD_05_PREFIX,
    VAL_BEST_PREFIX,
    build_train_command,
    experiment_feature_selection,
    generate_split_files,
    paired_delta_summary,
    run_experiment,
    split_path_for_seed,
    validate_experiment_selection,
    validate_custom_dataset_metadata,
    validate_paper_dataset_metadata,
    validate_split_files,
)


def _paper_data(endpoint_order: str = 'random', feature_group: str = 'paper14') -> Data:
    data = Data(x=torch.zeros(2, len(PAPER14_FEATURE_NAMES)))
    data.file_path = 'mesh_0.obj'
    data.feature_group = feature_group
    data.feature_preset = 'paper14'
    data.feature_names = list(PAPER14_FEATURE_NAMES)
    data.endpoint_order = endpoint_order
    return data


def _custom_data(feature_names: list[str] | None = None, endpoint_order: str = 'random') -> Data:
    names = feature_names or list(resolve_feature_selection(
        'custom',
        enable_ao=True,
        enable_dihedral=True,
        enable_symmetry=True,
        enable_density=True,
    ).feature_names)
    data = Data(x=torch.zeros(2, len(names)))
    data.file_path = 'mesh_0.obj'
    data.feature_group = 'custom'
    data.feature_preset = 'custom'
    data.feature_names = names
    data.endpoint_order = endpoint_order
    return data


def _summary(seed: int, fpr_best: float, f1_best: float, fpr_05: float, f1_05: float) -> dict:
    return {
        'best_epoch': seed + 1,
        'best_validation_threshold': 0.6,
        'resolution_selector': 'all',
        'filtered_graph_count': 4,
        'test_metrics_threshold_0_5': {
            'f1': f1_05,
            'precision': 0.4,
            'recall': 0.5,
            'fpr': fpr_05,
            'tpr': 0.5,
            'accuracy': 0.6,
        },
        'test_metrics_best_validation_threshold': {
            'f1': f1_best,
            'precision': 0.5,
            'recall': 0.6,
            'fpr': fpr_best,
            'tpr': 0.6,
            'accuracy': 0.7,
        },
    }


class FeatureAblationRunnerTests(unittest.TestCase):
    def test_experiment_name_to_feature_selection_mapping(self):
        control = experiment_feature_selection('custom14_control')
        extended_equiv = experiment_feature_selection('extended18_equiv')
        full_custom = experiment_feature_selection('full_custom')

        self.assertEqual(control.feature_group, 'custom')
        self.assertEqual(control.feature_names, PAPER14_FEATURE_NAMES)
        self.assertTrue(extended_equiv.feature_flags.ao)
        self.assertTrue(extended_equiv.feature_flags.signed_dihedral)
        self.assertTrue(extended_equiv.feature_flags.symmetry)
        self.assertFalse(extended_equiv.feature_flags.density)
        self.assertEqual(extended_equiv.feature_count, 18)
        self.assertTrue(full_custom.feature_flags.density)
        self.assertEqual(full_custom.feature_names[-2:], ('density_mean', 'density_diff'))

    def test_endpoint_order_safety_checks(self):
        validate_paper_dataset_metadata([_paper_data()])
        validate_custom_dataset_metadata([_custom_data()], ['full_custom'])

        with self.assertRaisesRegex(ValueError, "endpoint_order must be 'random'"):
            validate_paper_dataset_metadata([_paper_data(endpoint_order='fixed')])
        with self.assertRaisesRegex(ValueError, "endpoint_order must be 'random'"):
            validate_custom_dataset_metadata([_custom_data(endpoint_order='fixed')], ['custom14_control'])

    def test_failure_on_missing_features_or_wrong_dataset_metadata(self):
        with self.assertRaisesRegex(ValueError, "feature_group must be 'paper14'"):
            validate_paper_dataset_metadata([_paper_data(feature_group='extended18')])

        with self.assertRaisesRegex(ValueError, 'missing requested feature'):
            validate_custom_dataset_metadata([_custom_data(list(PAPER14_FEATURE_NAMES))], ['full_custom'])

    def test_ablation_selection_requires_custom_control(self):
        validate_experiment_selection(['paper14_locked'])
        validate_experiment_selection(['custom14_control'])
        validate_experiment_selection(['custom14_control', 'ao_only'])

        with self.assertRaisesRegex(ValueError, 'custom14_control'):
            validate_experiment_selection(['ao_only'])

    def test_split_generation_and_validation_reuse_dataset_agnostic_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / 'splits'
            dataset = [_custom_data() for _ in range(6)]
            for idx, data in enumerate(dataset):
                data.file_path = f'mesh_{idx}.obj'

            generate_split_files(
                source_dataset=dataset,
                splits_dir=splits_dir,
                seeds=[11, 12],
                group_mode='legacy',
                resolution_tag='all',
                val_ratio=0.2,
                test_ratio=0.2,
            )

            payload = json.loads(split_path_for_seed(splits_dir, 11).read_text())
            self.assertIsNone(payload['dataset_path'])
            args = Namespace(seeds=[11, 12], splits_dir=str(splits_dir), group_mode='legacy', resolution_tag='all')
            validate_split_files(args, {'custom': dataset})

    def test_split_validation_rejects_dataset_tied_split_files(self):
        with TemporaryDirectory() as tmp:
            split_path = Path(tmp) / 'splits' / 'seed_3.json'
            split_path.parent.mkdir()
            split_path.write_text(json.dumps({
                'train_group_ids': ['a'],
                'val_group_ids': ['b'],
                'test_group_ids': ['c'],
                'seed': 3,
                'group_mode': 'family',
                'dataset_path': '/tmp/custom.pt',
                'resolution_tag': 'all',
            }))
            args = Namespace(seeds=[3], splits_dir=str(split_path.parent), group_mode='family', resolution_tag='all')

            with self.assertRaisesRegex(ValueError, 'dataset-agnostic'):
                validate_split_files(args, {})

    def test_paired_delta_aggregation(self):
        experiment_records = [
            {
                'seed': 1,
                'status': 'completed',
                f'{VAL_BEST_PREFIX}_fpr': 0.08,
                f'{VAL_BEST_PREFIX}_recall': 0.62,
                f'{VAL_BEST_PREFIX}_f1': 0.56,
                f'{VAL_BEST_PREFIX}_accuracy': 0.72,
                f'{THRESHOLD_05_PREFIX}_fpr': 0.12,
                f'{THRESHOLD_05_PREFIX}_recall': 0.52,
                f'{THRESHOLD_05_PREFIX}_f1': 0.46,
                f'{THRESHOLD_05_PREFIX}_accuracy': 0.68,
            },
            {
                'seed': 2,
                'status': 'completed',
                f'{VAL_BEST_PREFIX}_fpr': 0.11,
                f'{VAL_BEST_PREFIX}_recall': 0.58,
                f'{VAL_BEST_PREFIX}_f1': 0.52,
                f'{VAL_BEST_PREFIX}_accuracy': 0.70,
                f'{THRESHOLD_05_PREFIX}_fpr': 0.14,
                f'{THRESHOLD_05_PREFIX}_recall': 0.50,
                f'{THRESHOLD_05_PREFIX}_f1': 0.44,
                f'{THRESHOLD_05_PREFIX}_accuracy': 0.66,
            },
        ]
        control_records = [
            {
                'seed': 1,
                'status': 'completed',
                f'{VAL_BEST_PREFIX}_fpr': 0.10,
                f'{VAL_BEST_PREFIX}_recall': 0.60,
                f'{VAL_BEST_PREFIX}_f1': 0.55,
                f'{VAL_BEST_PREFIX}_accuracy': 0.71,
                f'{THRESHOLD_05_PREFIX}_fpr': 0.11,
                f'{THRESHOLD_05_PREFIX}_recall': 0.50,
                f'{THRESHOLD_05_PREFIX}_f1': 0.45,
                f'{THRESHOLD_05_PREFIX}_accuracy': 0.67,
            },
            {
                'seed': 2,
                'status': 'completed',
                f'{VAL_BEST_PREFIX}_fpr': 0.10,
                f'{VAL_BEST_PREFIX}_recall': 0.60,
                f'{VAL_BEST_PREFIX}_f1': 0.53,
                f'{VAL_BEST_PREFIX}_accuracy': 0.71,
                f'{THRESHOLD_05_PREFIX}_fpr': 0.15,
                f'{THRESHOLD_05_PREFIX}_recall': 0.51,
                f'{THRESHOLD_05_PREFIX}_f1': 0.43,
                f'{THRESHOLD_05_PREFIX}_accuracy': 0.65,
            },
        ]

        delta = paired_delta_summary(
            experiment_name='ao_only',
            experiment_records=experiment_records,
            control_name='custom14_control',
            control_records=control_records,
        )

        self.assertEqual(delta['paired_seed_count'], 2)
        self.assertAlmostEqual(delta['val_best']['delta_test_val_best_fpr']['mean'], -0.005)
        self.assertEqual(delta['val_best']['win_count_fpr'], 1)
        self.assertEqual(delta['val_best']['win_count_f1'], 1)
        self.assertIn('threshold_0_5_diagnostics', delta)

    def test_subprocess_command_construction(self):
        paper_command = build_train_command(
            spec=EXPERIMENT_SPECS['paper14_locked'],
            paper_dataset='paper.pt',
            custom_dataset='custom.pt',
            run_dir=Path('runs') / 'paper',
            split_json=Path('splits') / 'seed_7.json',
            seed=7,
            resolution_tag='all',
            group_mode='family',
            epochs=3,
        )
        custom_command = build_train_command(
            spec=EXPERIMENT_SPECS['full_custom'],
            paper_dataset='paper.pt',
            custom_dataset='custom.pt',
            run_dir=Path('runs') / 'full',
            split_json=Path('splits') / 'seed_7.json',
            seed=7,
            resolution_tag='all',
            group_mode='family',
            epochs=3,
        )

        self.assertEqual(paper_command[0], sys.executable)
        self.assertIn(str(Path('tools') / 'run_baseline.py'), paper_command)
        self.assertIn('--split-json-in', paper_command)
        self.assertNotIn('--split-json-out', paper_command)
        self.assertIn('--strict-paper-protocol', paper_command)
        self.assertNotIn('--strict-paper-protocol', custom_command)
        self.assertIn('--enable-ao', custom_command)
        self.assertIn('--enable-dihedral', custom_command)
        self.assertIn('--enable-symmetry', custom_command)
        self.assertIn('--enable-density', custom_command)

    def test_run_experiment_reuses_existing_split_jsons(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / 'splits'
            splits_dir.mkdir()
            for seed in [1, 2]:
                split_path_for_seed(splits_dir, seed).write_text('{}')

            commands = []

            def fake_runner(command, check):
                commands.append(command)
                seed = int(command[command.index('--seed') + 1])
                run_dir = Path(command[command.index('--run-dir') + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / 'summary.json').write_text(json.dumps(_summary(seed, 0.1, 0.5, 0.2, 0.4)))

            args = Namespace(
                paper_dataset='paper.pt',
                custom_dataset='custom.pt',
                output_root=str(root),
                splits_dir=str(splits_dir),
                seeds=[1, 2],
                resolution_tag='all',
                group_mode='family',
                epochs=1,
                keep_going=False,
            )

            spec = EXPERIMENT_SPECS['custom14_control']
            records = run_experiment(args=args, spec=spec, runner=fake_runner)

        self.assertEqual([record['status'] for record in records], ['completed', 'completed'])
        self.assertEqual(commands[0][commands[0].index('--split-json-in') + 1], str(splits_dir / 'seed_1.json'))
        self.assertEqual(commands[1][commands[1].index('--split-json-in') + 1], str(splits_dir / 'seed_2.json'))
        self.assertNotIn('--split-json-out', commands[0])

    def test_run_experiment_records_subprocess_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                paper_dataset='paper.pt',
                custom_dataset='custom.pt',
                output_root=str(root),
                splits_dir=str(root / 'splits'),
                seeds=[1, 2],
                resolution_tag='all',
                group_mode='family',
                epochs=1,
                keep_going=False,
            )
            spec = EXPERIMENT_SPECS['custom14_control']

            def fake_runner(command, check):
                raise subprocess.CalledProcessError(9, command)

            records = run_experiment(args=args, spec=spec, runner=fake_runner)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['status'], 'failed')
        self.assertIn('baseline runner exited with 9', records[0]['error'])


if __name__ == '__main__':
    unittest.main()
