import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from torch_geometric.data import Data

from models.utils.dataset import filter_dataset_by_resolution, infer_resolution_selector, split_dataset
from models.utils.filename_parsing import parse_mesh_name


def _data(file_path):
    data = Data()
    data.file_path = file_path
    return data


class DatasetGroupingTests(unittest.TestCase):
    def test_default_grouping_uses_family_ids(self):
        dataset = [
            _data('mesh_10000f.obj'),
            _data('mesh_8000f.obj'),
            _data('other_aug0.obj'),
        ]

        _, _, _, split_info = split_dataset(dataset, val_ratio=0.34, test_ratio=0.34, seed=1)
        keys = set(split_info['train'] + split_info['val'] + split_info['test'])

        self.assertIn('mesh', keys)
        self.assertIn('other', keys)

    def test_family_grouping_combines_resolution_variants(self):
        dataset = [
            _data('mesh_10000f.obj'),
            _data('mesh_8000f_aug1.obj'),
            _data('other_aug0.obj'),
        ]

        _, _, _, split_info = split_dataset(
            dataset,
            val_ratio=0.34,
            test_ratio=0.34,
            seed=1,
            group_mode='family',
        )
        keys = set(split_info['train'] + split_info['val'] + split_info['test'])

        self.assertIn('mesh', keys)
        self.assertIn('other', keys)
        self.assertNotIn('mesh_10000f', keys)

    def test_family_grouping_combines_native_resolution_and_aug_variants(self):
        dataset = [
            _data('man013.obj'),
            _data('man013_aug0.obj'),
            _data('man013_l.obj'),
            _data('man013_l_aug0.obj'),
            _data('man013_h.obj'),
            _data('man013_h_aug1.obj'),
            _data('other.obj'),
        ]

        _, _, _, split_info = split_dataset(
            dataset,
            val_ratio=0.3,
            test_ratio=0.3,
            seed=4,
            group_mode='family',
        )
        keys = set(split_info['train'] + split_info['val'] + split_info['test'])

        self.assertEqual(keys, {'man013', 'other'})

    def test_filter_dataset_by_resolution(self):
        dataset = [
            _data('mesh_10000f.obj'),
            _data('mesh_8000f.obj'),
            _data('other.obj'),
        ]

        filtered = filter_dataset_by_resolution(dataset, '10000f')

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].file_path, 'mesh_10000f.obj')

    def test_filter_dataset_by_native_resolution_selectors(self):
        dataset = [
            _data('fem001.obj'),
            _data('fem001_h.obj'),
            _data('fem001_l.obj'),
            _data('fem002_aug1.obj'),
        ]

        self.assertEqual([d.file_path for d in filter_dataset_by_resolution(dataset, 'all')], [
            'fem001.obj',
            'fem001_h.obj',
            'fem001_l.obj',
            'fem002_aug1.obj',
        ])
        self.assertEqual([d.file_path for d in filter_dataset_by_resolution(dataset, 'base')], [
            'fem001.obj',
            'fem002_aug1.obj',
        ])
        self.assertEqual([d.file_path for d in filter_dataset_by_resolution(dataset, 'h')], ['fem001_h.obj'])
        self.assertEqual([d.file_path for d in filter_dataset_by_resolution(dataset, 'l')], ['fem001_l.obj'])

    def test_infer_resolution_selector_preserves_custom_tags(self):
        self.assertEqual(infer_resolution_selector('mesh_10000f.obj'), '10000f')
        self.assertEqual(infer_resolution_selector('mesh_res12_aug1.obj'), 'res12')

    def test_empty_resolution_filter_reports_requested_and_available_selectors(self):
        dataset = [_data('fem001.obj'), _data('fem001_h.obj')]

        with self.assertRaisesRegex(ValueError, "resolution selector 'l'.*available selectors: base, h"):
            filter_dataset_by_resolution(dataset, 'l')

    def test_split_save_load_roundtrip_preserves_groups(self):
        dataset = [_data(f'mesh_{idx}_10000f.obj') for idx in range(6)]

        with TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / 'dataset_dual.pt'
            split_path = Path(tmp) / 'split.json'

            _, _, _, saved_info = split_dataset(
                dataset,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=123,
                group_mode='family',
                split_json_out=split_path,
                dataset_path=dataset_path,
                resolution_tag='10000f',
            )
            _, _, _, loaded_info = split_dataset(
                dataset,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=999,
                group_mode='family',
                split_json_in=split_path,
                dataset_path=dataset_path,
                resolution_tag='10000f',
            )

        self.assertEqual(saved_info['train'], loaded_info['train'])
        self.assertEqual(saved_info['val'], loaded_info['val'])
        self.assertEqual(saved_info['test'], loaded_info['test'])

    def test_split_dataset_rejects_removed_group_mode(self):
        dataset = [_data('mesh_10000f.obj'), _data('other.obj')]

        with self.assertRaisesRegex(ValueError, "group_mode must be 'family'"):
            split_dataset(dataset, group_mode='legacy')

    def test_saved_split_json_schema_keys_are_stable(self):
        dataset = [_data(f'mesh_{idx}_10000f.obj') for idx in range(5)]

        with TemporaryDirectory() as tmp:
            split_path = Path(tmp) / 'split.json'
            split_dataset(
                dataset,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=123,
                group_mode='family',
                split_json_out=split_path,
                dataset_path=Path(tmp) / 'dataset_dual.pt',
                resolution_tag='10000f',
            )

            with open(split_path) as f:
                payload = json.load(f)

        self.assertEqual(set(payload), {
            'train_group_ids',
            'val_group_ids',
            'test_group_ids',
            'seed',
            'group_mode',
            'dataset_path',
            'resolution_tag',
        })

    def test_loaded_split_json_with_overlapping_groups_raises(self):
        dataset = [
            _data('mesh_h.obj'),
            _data('mesh_l.obj'),
            _data('other.obj'),
        ]

        with TemporaryDirectory() as tmp:
            split_path = Path(tmp) / 'split.json'
            with open(split_path, 'w') as f:
                json.dump({
                    'train_group_ids': ['mesh'],
                    'val_group_ids': ['mesh'],
                    'test_group_ids': ['other'],
                    'seed': 1,
                    'group_mode': 'family',
                    'dataset_path': None,
                    'resolution_tag': None,
                }, f)

            with self.assertRaisesRegex(ValueError, 'multiple splits'):
                split_dataset(
                    dataset,
                    val_ratio=0.2,
                    test_ratio=0.2,
                    seed=99,
                    group_mode='family',
                    split_json_in=split_path,
                )

    def test_generated_family_split_has_no_overlap(self):
        dataset = [
            _data('man013.obj'),
            _data('man013_l.obj'),
            _data('man013_h_aug0.obj'),
            _data('fem001.obj'),
            _data('fem001_h_aug0.obj'),
            _data('chair_10000f_aug2.obj'),
            _data('house_lod3_aug0.obj'),
        ]

        train, val, test, _ = split_dataset(
            dataset,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=11,
            group_mode='family',
        )

        split_families = [
            {parse_mesh_name(d.file_path).family_id for d in split}
            for split in (train, val, test)
        ]

        self.assertFalse(split_families[0] & split_families[1])
        self.assertFalse(split_families[0] & split_families[2])
        self.assertFalse(split_families[1] & split_families[2])

    def test_weighted_split_keeps_large_family_in_training(self):
        dataset = [_data(f'large_aug{idx}.obj') for idx in range(20)]
        dataset.extend([
            _data('small_a.obj'),
            _data('small_b.obj'),
            _data('small_c.obj'),
        ])

        train, val, test, split_info = split_dataset(
            dataset,
            val_ratio=0.34,
            test_ratio=0.34,
            seed=5,
            group_mode='family',
        )

        self.assertIn('large', split_info['train'])
        self.assertGreater(len(train), len(val))
        self.assertGreater(len(train), len(test))

    def test_seeded_split_generation_is_reproducible(self):
        dataset = [_data(f'mesh_{idx}_10000f.obj') for idx in range(8)]

        _, _, _, first = split_dataset(
            dataset,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=7,
            group_mode='family',
        )
        _, _, _, second = split_dataset(
            dataset,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=7,
            group_mode='family',
        )

        self.assertEqual(first['train'], second['train'])
        self.assertEqual(first['val'], second['val'])
        self.assertEqual(first['test'], second['test'])


if __name__ == '__main__':
    unittest.main()
