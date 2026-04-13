import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from torch_geometric.data import Data

from models.utils.dataset import filter_dataset_by_resolution, infer_resolution_selector, split_dataset


def _data(file_path):
    data = Data()
    data.file_path = file_path
    return data


class DatasetGroupingTests(unittest.TestCase):
    def test_default_grouping_keeps_resolution_variants_separate(self):
        dataset = [
            _data('mesh_10000f.obj'),
            _data('mesh_8000f.obj'),
            _data('other_aug0.obj'),
        ]

        _, _, _, split_info = split_dataset(dataset, val_ratio=0.34, test_ratio=0.34, seed=1)
        keys = set(split_info['train'] + split_info['val'] + split_info['test'])

        self.assertIn('mesh_10000f', keys)
        self.assertIn('mesh_8000f', keys)
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
                group_mode='legacy',
                split_json_out=split_path,
                dataset_path=dataset_path,
                resolution_tag='10000f',
            )
            _, _, _, loaded_info = split_dataset(
                dataset,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=999,
                group_mode='legacy',
                split_json_in=split_path,
                dataset_path=dataset_path,
                resolution_tag='10000f',
            )

        self.assertEqual(saved_info['train'], loaded_info['train'])
        self.assertEqual(saved_info['val'], loaded_info['val'])
        self.assertEqual(saved_info['test'], loaded_info['test'])

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
