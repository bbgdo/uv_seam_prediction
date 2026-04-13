import unittest
from pathlib import Path

from models.utils.filename_parsing import FilenameParseConfig, legacy_base_name, parse_mesh_name


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

    def test_custom_resolution_suffix(self):
        config = FilenameParseConfig(resolution_patterns=(r'_lod\d+$',))
        info = parse_mesh_name('house_lod3_aug0.obj', config)

        self.assertEqual(info.family_id, 'house')
        self.assertEqual(info.resolution_tag, 'lod3')
        self.assertTrue(info.is_augmented)

    def test_legacy_base_name_keeps_resolution(self):
        self.assertEqual(legacy_base_name('chair_10000f_aug2.obj'), 'chair_10000f')


if __name__ == '__main__':
    unittest.main()
