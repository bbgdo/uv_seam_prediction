import unittest
from pathlib import Path

from models.utils.filename_parsing import parse_mesh_name


class FilenameParsingTests(unittest.TestCase):
    def test_strips_augmentation_and_resolution(self):
        info = parse_mesh_name('chair_10000f_aug2.obj')

        self.assertEqual(info.stem, 'chair_10000f_aug2')
        self.assertEqual(info.family_id, 'chair')
        self.assertEqual(info.resolution_tag, '10000f')
        self.assertTrue(info.is_augmented)

    def test_strips_res_suffix(self):
        info = parse_mesh_name(Path('folder/dragon_res12.obj'))

        self.assertEqual(info.family_id, 'dragon')
        self.assertEqual(info.resolution_tag, 'res12')
        self.assertFalse(info.is_augmented)

    def test_strips_native_resolution_suffixes(self):
        self.assertEqual(parse_mesh_name('man013_l.obj').family_id, 'man013')
        self.assertEqual(parse_mesh_name('man013_h_aug0.obj').family_id, 'man013')

    def test_windows_separator_in_name(self):
        info = parse_mesh_name('3d-objs\\fem001_h_aug0.obj')

        self.assertEqual(info.stem, 'fem001_h_aug0')
        self.assertEqual(info.family_id, 'fem001')
        self.assertEqual(info.resolution_tag, 'h')
        self.assertTrue(info.is_augmented)

    def test_strips_repeated_resolution_and_augmentation_suffixes(self):
        self.assertEqual(parse_mesh_name('chair_10000f_aug2.obj').family_id, 'chair')
        self.assertEqual(parse_mesh_name('house_lod3_aug0.obj').family_id, 'house')

if __name__ == '__main__':
    unittest.main()
