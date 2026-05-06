import json
import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from torch_geometric.data import Data

from models.meshcnn_full.mesh import MeshCNNSample
from preprocessing.feature_registry import PAPER14_FEATURE_NAMES, resolve_feature_selection
from tools.run_feature_ablations import (
    ALL_COMBINATORIAL_SUITE,
    ALL_EXPERIMENT_SPECS,
    EXPERIMENT_SPECS,
    FULL_ABLATION_SUITE,
    THRESHOLD_05_PREFIX,
    VAL_BEST_PREFIX,
    build_experiment_payload,
    build_train_command,
    experiment_feature_selection,
    generate_split_files,
    paired_delta_summary,
    parse_args,
    run_experiment,
    run_suite,
    split_path_for_seed,
    validate_dataset_roles,
    validate_experiment_selection,
    validate_custom_dataset_metadata,
    validate_meshcnn_dataset_metadata,
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
        enable_thickness_sdf=True,
    ).feature_names)
    data = Data(x=torch.zeros(2, len(names)))
    data.file_path = 'mesh_0.obj'
    data.feature_group = 'custom'
    data.feature_preset = 'custom'
    data.feature_names = names
    data.endpoint_order = endpoint_order
    return data


def _meshcnn_sample(feature_names: list[str] | None = None, endpoint_order: str = 'random') -> MeshCNNSample:
    names = feature_names or list(resolve_feature_selection(
        'custom',
        enable_ao=True,
        enable_dihedral=True,
        enable_symmetry=True,
        enable_density=True,
        enable_thickness_sdf=True,
    ).feature_names)
    edge_count = 3
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    return MeshCNNSample(
        vertices=torch.zeros(3, 3),
        faces=faces,
        unique_edges=torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.long),
        edge_features=torch.zeros(edge_count, len(names)),
        edge_labels=torch.zeros(edge_count),
        edge_neighbors=torch.full((edge_count, 4), -1, dtype=torch.long),
        edge_to_faces=torch.full((edge_count, 2), -1, dtype=torch.long),
        face_to_edges=torch.zeros(1, 3, dtype=torch.long),
        boundary_mask=torch.ones(edge_count, dtype=torch.bool),
        file_path='mesh_0.obj',
        feature_group='custom',
        feature_preset='custom',
        feature_names=names,
        feature_flags={},
        endpoint_order=endpoint_order,
        label_source='exact_obj',
    )


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


_EXPECTED_ORDER = [
    'control14',
    'ao', 'sdf', 'dihedral', 'symmetry', 'density',
    'ao_sdf', 'ao_dihedral', 'ao_symmetry', 'ao_density',
    'sdf_dihedral', 'sdf_symmetry', 'sdf_density',
    'dihedral_symmetry', 'dihedral_density', 'symmetry_density',
]


class FeatureAblationRunnerTests(unittest.TestCase):
    def test_experiment_suite_structure(self):
        self.assertEqual(len(EXPERIMENT_SPECS), 16)
        self.assertEqual(len(ALL_EXPERIMENT_SPECS), 32)
        self.assertIn('control14', EXPERIMENT_SPECS)
        self.assertNotIn('full_custom', EXPERIMENT_SPECS)
        self.assertNotIn('full_custom_sdf', EXPERIMENT_SPECS)
        self.assertNotIn('ao_dihedral_symmetry', EXPERIMENT_SPECS)
        self.assertNotIn('ao_density_sdf', EXPERIMENT_SPECS)
        self.assertNotIn('ao_dihedral_symmetry_density_sdf', EXPERIMENT_SPECS)
        self.assertIn('ao_dihedral_symmetry', ALL_EXPERIMENT_SPECS)
        self.assertIn('ao_sdf_density', ALL_EXPERIMENT_SPECS)
        self.assertIn('ao_sdf_dihedral_symmetry_density', ALL_EXPERIMENT_SPECS)

    def test_every_key_equals_spec_name(self):
        for key, spec in EXPERIMENT_SPECS.items():
            self.assertEqual(key, spec.name, f"key {key!r} != spec.name {spec.name!r}")

    def test_all_experiments_use_custom_feature_group(self):
        for name, spec in EXPERIMENT_SPECS.items():
            expected = 'paper14' if name == 'control14' else 'custom'
            self.assertEqual(spec.feature_group, expected, f"{name}: wrong feature_group")

    def test_experiment_order(self):
        self.assertEqual(list(EXPERIMENT_SPECS.keys()), _EXPECTED_ORDER)
        self.assertEqual(list(FULL_ABLATION_SUITE), _EXPECTED_ORDER)
        self.assertEqual(list(ALL_COMBINATORIAL_SUITE[:16]), _EXPECTED_ORDER)

    def test_experiment_name_to_feature_selection_mapping(self):
        control = experiment_feature_selection('control14')
        ao_dihedral = experiment_feature_selection('ao_dihedral')
        sdf_density = experiment_feature_selection('sdf_density')

        self.assertEqual(control.feature_group, 'paper14')
        self.assertEqual(control.feature_names, PAPER14_FEATURE_NAMES)

        self.assertTrue(ao_dihedral.feature_flags.ao)
        self.assertTrue(ao_dihedral.feature_flags.signed_dihedral)
        self.assertFalse(ao_dihedral.feature_flags.symmetry)
        self.assertFalse(ao_dihedral.feature_flags.density)
        self.assertEqual(ao_dihedral.feature_count, 17)

        self.assertTrue(sdf_density.feature_flags.density)
        self.assertTrue(sdf_density.feature_flags.thickness_sdf)
        self.assertEqual(sdf_density.feature_names[-3:], ('density_mean', 'density_diff', 'thickness_sdf'))

    def test_endpoint_order_safety_checks(self):
        validate_custom_dataset_metadata([_custom_data()], ['ao_dihedral'])
        validate_custom_dataset_metadata([_custom_data(endpoint_order='fixed')], ['control14'])
        validate_meshcnn_dataset_metadata([_meshcnn_sample(endpoint_order='fixed')], ['control14'])

        with self.assertRaisesRegex(ValueError, "endpoint_order must be one of"):
            validate_custom_dataset_metadata([_custom_data(endpoint_order='invalid')], ['control14'])

    def test_failure_on_missing_features_or_wrong_dataset_metadata(self):
        with self.assertRaisesRegex(ValueError, 'missing requested feature'):
            validate_custom_dataset_metadata(
                [_custom_data(list(PAPER14_FEATURE_NAMES))],
                ['ao_density'],
            )
        without_sdf = resolve_feature_selection(
            'custom',
            enable_ao=True,
            enable_dihedral=True,
            enable_symmetry=True,
            enable_density=True,
        ).feature_names
        with self.assertRaisesRegex(ValueError, 'thickness_sdf'):
            validate_custom_dataset_metadata(
                [_custom_data(list(without_sdf))],
                ['ao_sdf'],
            )

    def test_validate_experiment_selection_accepts_all_specs(self):
        validate_experiment_selection(list(EXPERIMENT_SPECS), model='graphsage')
        validate_experiment_selection(list(EXPERIMENT_SPECS), model='gatv2')
        validate_experiment_selection(list(EXPERIMENT_SPECS), model='sparsemeshcnn')
        with self.assertRaisesRegex(ValueError, 'unsupported ablation model'):
            validate_experiment_selection(list(EXPERIMENT_SPECS), model='meshcnn_full')

    def test_parse_args_accepts_sparsemeshcnn_and_rejects_internal_meshcnn_name(self):
        args = parse_args([
            '--model', 'sparsemeshcnn',
            '--meshcnn-dataset', 'meshcnn.pt',
            '--output-root', 'out',
        ])
        self.assertEqual(args.model, 'sparsemeshcnn')
        self.assertEqual(args.seeds, [33])
        self.assertEqual(args.epochs, 60)
        self.assertEqual(args.patience, 15)
        self.assertIsNone(args.split_json_in)

        split_args = parse_args([
            '--model', 'graphsage',
            '--gnn-dataset', 'custom.pt',
            '--split-json-in', 'splits/fixed.json',
            '--output-root', 'out',
        ])
        self.assertEqual(split_args.split_json_in, 'splits/fixed.json')

        full_args = parse_args([
            '--model', 'graphsage',
            '--gnn-dataset', 'custom.pt',
            '--combinatorial-suite', '1', '2', '3', '4', '5',
            '--output-root', 'out',
        ])
        self.assertEqual(len(full_args.experiments), 32)
        self.assertIn('ao_sdf_dihedral_symmetry_density', full_args.experiments)

        pairwise_args = parse_args([
            '--model', 'graphsage',
            '--gnn-dataset', 'custom.pt',
            '--combinatorial-suite', '1', '2',
            '--output-root', 'out',
        ])
        self.assertEqual(pairwise_args.experiments, list(FULL_ABLATION_SUITE))

        with self.assertRaises(SystemExit):
            parse_args([
                '--model', 'graphsage',
                '--custom-dataset', 'custom.pt',
                '--output-root', 'out',
            ])

        with self.assertRaises(SystemExit):
            parse_args([
                '--model', 'meshcnn_full',
                '--meshcnn-dataset', 'meshcnn.pt',
                '--output-root', 'out',
            ])

    def test_meshcnn_dataset_validation_uses_superset_features(self):
        validate_meshcnn_dataset_metadata([_meshcnn_sample()], ['control14', 'ao_sdf'])
        with self.assertRaisesRegex(ValueError, 'thickness_sdf'):
            validate_meshcnn_dataset_metadata([_meshcnn_sample(list(PAPER14_FEATURE_NAMES))], ['sdf'])

    def test_meshcnn_dataset_is_required_without_gnn_dataset(self):
        args = Namespace(
            model='sparsemeshcnn',
            meshcnn_dataset=None,
            resolution_tag='all',
        )
        with self.assertRaisesRegex(ValueError, '--meshcnn-dataset is required'):
            validate_dataset_roles(args, ['control14'])

        args.meshcnn_dataset = 'meshcnn.pt'
        with mock.patch(
            'tools.run_feature_ablations.load_filtered_meshcnn_dataset',
            return_value=[_meshcnn_sample()],
        ):
            datasets = validate_dataset_roles(args, ['control14'])
        self.assertIn('meshcnn', datasets)

    def test_meshcnn_generate_splits_uses_meshcnn_dataset(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = [_meshcnn_sample() for _ in range(4)]
            for idx, sample in enumerate(samples):
                sample.file_path = f'mesh_{idx}.obj'
            args = Namespace(
                model='sparsemeshcnn',
                meshcnn_dataset='meshcnn.pt',
                experiments=['control14'],
                output_root=str(root),
                splits_dir=str(root / 'splits'),
                seeds=[3],
                resolution_tag='all',
                epochs=1,
                generate_splits=True,
                only_generate_splits=True,
                val_ratio=0.25,
                test_ratio=0.25,
                keep_going=False,
            )

            with mock.patch(
                'tools.run_feature_ablations.load_filtered_meshcnn_dataset',
                return_value=samples,
            ):
                payloads = run_suite(args)

            self.assertEqual(payloads, {})
            self.assertTrue(split_path_for_seed(Path(args.splits_dir), 33).exists())

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
                resolution_tag='all',
                val_ratio=0.2,
                test_ratio=0.2,
            )

            payload = json.loads(split_path_for_seed(splits_dir, 11).read_text())
            self.assertIsNone(payload['dataset_path'])
            args = Namespace(seeds=[11, 12], splits_dir=str(splits_dir), resolution_tag='all')
            validate_split_files(args, {'custom': dataset})

    def test_split_json_in_overrides_splits_dir_for_validation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated_dir = root / 'generated'
            dataset = [_custom_data() for _ in range(6)]
            for idx, data in enumerate(dataset):
                data.file_path = f'mesh_{idx}.obj'

            generate_split_files(
                source_dataset=dataset,
                splits_dir=generated_dir,
                seeds=[33],
                resolution_tag='all',
                val_ratio=0.2,
                test_ratio=0.2,
            )

            explicit_split = root / 'fixed_split.json'
            explicit_split.write_text(split_path_for_seed(generated_dir, 33).read_text())
            args = Namespace(
                seeds=[33],
                splits_dir=str(root / 'unused_splits_dir'),
                split_json_in=str(explicit_split),
                resolution_tag='all',
            )
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
            args = Namespace(seeds=[3], splits_dir=str(split_path.parent), resolution_tag='all')

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
            experiment_name='ao',
            experiment_records=experiment_records,
            control_name='control14',
            control_records=control_records,
        )

        self.assertEqual(delta['paired_seed_count'], 2)
        self.assertAlmostEqual(delta['val_best']['delta_test_val_best_fpr']['mean'], -0.005)
        self.assertEqual(delta['val_best']['win_count_fpr'], 1)
        self.assertEqual(delta['val_best']['win_count_f1'], 1)
        self.assertIn('threshold_0_5_diagnostics', delta)

    def test_subprocess_command_construction(self):
        custom_command = build_train_command(
            spec=EXPERIMENT_SPECS['ao_dihedral'],
            dataset='custom.pt',
            run_dir=Path('runs') / 'full',
            split_json=Path('splits') / 'seed_7.json',
            seed=7,
            resolution_tag='all',
            epochs=3,
        )
        gatv2_command = build_train_command(
            spec=EXPERIMENT_SPECS['control14'],
            dataset='custom.pt',
            run_dir=Path('runs') / 'gatv2',
            split_json=Path('splits') / 'seed_7.json',
            seed=7,
            resolution_tag='all',
            epochs=3,
            model='gatv2',
        )
        sdf_command = build_train_command(
            spec=EXPERIMENT_SPECS['sdf_density'],
            dataset='custom.pt',
            run_dir=Path('runs') / 'sdf',
            split_json=Path('splits') / 'seed_7.json',
            seed=7,
            resolution_tag='all',
            epochs=3,
            model='gatv2',
        )

        self.assertEqual(custom_command[0], sys.executable)
        self.assertIn(str(Path('tools') / 'run_baseline.py'), custom_command)
        self.assertIn('--split-json-in', custom_command)
        self.assertNotIn('--split-json-out', custom_command)
        self.assertIn('--enable-ao', custom_command)
        self.assertIn('--enable-dihedral', custom_command)
        self.assertNotIn('--enable-symmetry', custom_command)
        self.assertNotIn('--preset', custom_command)
        self.assertNotIn('--pos-weight', custom_command)

        self.assertEqual(gatv2_command[gatv2_command.index('--model') + 1], 'gatv2')
        self.assertNotIn('--preset', gatv2_command)
        self.assertNotIn('--pos-weight', gatv2_command)

        self.assertEqual(sdf_command[sdf_command.index('--model') + 1], 'gatv2')
        self.assertIn('--enable-density', sdf_command)
        self.assertIn('--enable-thickness-sdf', sdf_command)
        self.assertNotIn('--enable-ao', sdf_command)
        self.assertNotIn('--pos-weight', sdf_command)

    def test_meshcnn_subprocess_command_construction(self):
        command = build_train_command(
            spec=EXPERIMENT_SPECS['ao_sdf'],
            dataset=None,
            meshcnn_dataset='meshcnn_superset.pt',
            run_dir=Path('out') / 'sparsemeshcnn' / 'experiments' / 'ao_sdf' / 'seed_7',
            split_json=Path('splits') / 'seed_7.json',
            seed=7,
            resolution_tag='all',
            epochs=3,
            model='sparsemeshcnn',
        )

        self.assertEqual(command[0], sys.executable)
        self.assertIn(str(Path('models') / 'meshcnn_full' / 'train.py'), command)
        self.assertNotIn(str(Path('tools') / 'run_baseline.py'), command)
        for flag in ('--dataset', '--run-dir', '--epochs', '--seed', '--split-json-in'):
            self.assertIn(flag, command)
        self.assertNotIn('--group-mode', command)
        self.assertEqual(command[command.index('--dataset') + 1], 'meshcnn_superset.pt')
        self.assertEqual(command[command.index('--feature-group') + 1], 'custom')
        self.assertIn('--enable-ao', command)
        self.assertIn('--enable-thickness-sdf', command)
        self.assertNotIn('--enable-dihedral', command)
        self.assertNotIn('--enable-symmetry', command)
        self.assertNotIn('--enable-density', command)
        for forbidden in ('--model', '--preset', '--pos-weight'):
            self.assertNotIn(forbidden, command)

    def test_run_experiment_reuses_existing_split_jsons(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / 'splits'
            splits_dir.mkdir()
            for seed in [33]:
                split_path_for_seed(splits_dir, seed).write_text('{}')

            commands = []

            def fake_runner(command, check):
                commands.append(command)
                seed = int(command[command.index('--seed') + 1])
                run_dir = Path(command[command.index('--run-dir') + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / 'summary.json').write_text(json.dumps(_summary(seed, 0.1, 0.5, 0.2, 0.4)))

            args = Namespace(
                gnn_dataset='custom.pt',
                meshcnn_dataset=None,
                output_root=str(root),
                splits_dir=str(splits_dir),
                seeds=[1, 2],
                resolution_tag='all',
                epochs=1,
                keep_going=False,
                model='gatv2',
            )

            spec = EXPERIMENT_SPECS['control14']
            records = run_experiment(args=args, spec=spec, runner=fake_runner)

        self.assertEqual([record['status'] for record in records], ['completed'])
        self.assertEqual(
            records[0]['run_dir'],
            str(root / 'gatv2' / 'experiments' / 'control14' / 'seed_33'),
        )
        self.assertEqual(commands[0][commands[0].index('--split-json-in') + 1], str(splits_dir / 'seed_33.json'))
        self.assertEqual(commands[0][commands[0].index('--model') + 1], 'gatv2')
        self.assertNotIn('--split-json-out', commands[0])

    def test_run_experiment_uses_explicit_split_json_in(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit_split = root / 'fixed_split.json'
            explicit_split.write_text('{}')
            commands = []

            def fake_runner(command, check):
                commands.append(command)
                run_dir = Path(command[command.index('--run-dir') + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / 'summary.json').write_text(json.dumps(_summary(33, 0.1, 0.5, 0.2, 0.4)))

            args = Namespace(
                gnn_dataset='custom.pt',
                meshcnn_dataset=None,
                output_root=str(root),
                splits_dir=str(root / 'unused_splits_dir'),
                split_json_in=str(explicit_split),
                seeds=[33],
                resolution_tag='all',
                epochs=1,
                keep_going=False,
                model='gatv2',
            )

            records = run_experiment(args=args, spec=EXPERIMENT_SPECS['control14'], runner=fake_runner)

        self.assertEqual(records[0]['split_json'], str(explicit_split))
        self.assertEqual(commands[0][commands[0].index('--split-json-in') + 1], str(explicit_split))

    def test_run_experiment_records_subprocess_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                gnn_dataset='custom.pt',
                output_root=str(root),
                splits_dir=str(root / 'splits'),
                seeds=[33],
                resolution_tag='all',
                epochs=1,
                keep_going=False,
                model='graphsage',
            )
            spec = EXPERIMENT_SPECS['control14']

            def fake_runner(command, check):
                raise subprocess.CalledProcessError(9, command)

            records = run_experiment(args=args, spec=spec, runner=fake_runner)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['status'], 'failed')
        self.assertIn('train runner exited with 9', records[0]['error'])

    def test_run_experiment_uses_sparsemeshcnn_output_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / 'splits'
            splits_dir.mkdir()
            split_path_for_seed(splits_dir, 33).write_text('{}')
            commands = []

            def fake_runner(command, check):
                commands.append(command)
                run_dir = Path(command[command.index('--run-dir') + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / 'summary.json').write_text(json.dumps(_summary(1, 0.1, 0.5, 0.2, 0.4)))

            args = Namespace(
                meshcnn_dataset='meshcnn.pt',
                output_root=str(root),
                splits_dir=str(splits_dir),
                seeds=[1],
                resolution_tag='all',
                epochs=1,
                keep_going=False,
                model='sparsemeshcnn',
            )

            records = run_experiment(args=args, spec=EXPERIMENT_SPECS['control14'], runner=fake_runner)

        self.assertEqual(records[0]['status'], 'completed')
        self.assertEqual(
            records[0]['run_dir'],
            str(root / 'sparsemeshcnn' / 'experiments' / 'control14' / 'seed_33'),
        )
        self.assertIn(str(Path('models') / 'meshcnn_full' / 'train.py'), commands[0])

    def test_sparsemeshcnn_payload_reports_public_model_name(self):
        args = Namespace(
            model='sparsemeshcnn',
            meshcnn_dataset='meshcnn.pt',
            resolution_tag='all',
            epochs=1,
            seeds=[1],
            splits_dir='splits',
        )

        payload = build_experiment_payload(
            args=args,
            spec=EXPERIMENT_SPECS['control14'],
            records=[],
        )

        self.assertEqual(payload['model'], 'sparsemeshcnn')
        self.assertEqual(payload['dataset'], 'meshcnn.pt')
        self.assertNotIn('preset', payload)


if __name__ == '__main__':
    unittest.main()
