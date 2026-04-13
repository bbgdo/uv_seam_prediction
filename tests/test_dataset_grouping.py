import unittest

from torch_geometric.data import Data

from models.utils.dataset import filter_dataset_by_resolution, split_dataset


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


if __name__ == '__main__':
    unittest.main()
