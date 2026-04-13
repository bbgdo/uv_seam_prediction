import unittest

from preprocessing.obj_parser import ObjParseError, parse_obj_text


class ObjParserTests(unittest.TestCase):
    def test_parses_supported_face_token_forms(self):
        mesh = parse_obj_text(
            """
            v 0 0 0
            v 1 0 0
            v 0 1 0
            v 0 0 1
            vt 0 0
            vt 1 0
            vt 0 1
            vt 1 1
            vn 0 0 1
            f 1/1/1 2/2/1 3/3/1
            f 1/1 3/3 4/4
            f 1//1 4//1 2//1
            f 1 2 4
            """,
            file_path='inline.obj',
        )

        self.assertEqual(mesh.file_path, 'inline.obj')
        self.assertEqual(len(mesh.faces), 4)
        self.assertEqual(mesh.faces[0].line_number, 11)
        self.assertEqual(mesh.faces[0].corners[0].vertex_index, 0)
        self.assertEqual(mesh.faces[0].corners[0].uv_index, 0)
        self.assertEqual(mesh.faces[0].corners[0].normal_index, 0)
        self.assertIsNone(mesh.faces[1].corners[0].normal_index)
        self.assertIsNone(mesh.faces[2].corners[0].uv_index)
        self.assertIsNone(mesh.faces[3].corners[0].uv_index)
        self.assertIsNone(mesh.faces[3].corners[0].normal_index)

    def test_resolves_negative_indices(self):
        mesh = parse_obj_text(
            """
            v 0 0 0
            v 1 0 0
            v 0 1 0
            vt 0 0
            vt 1 0
            vt 0 1
            vn 0 0 1
            f -3/-3/-1 -2/-2/-1 -1/-1/-1
            """
        )

        face = mesh.faces[0]
        self.assertEqual([corner.vertex_index for corner in face.corners], [0, 1, 2])
        self.assertEqual([corner.uv_index for corner in face.corners], [0, 1, 2])
        self.assertEqual([corner.normal_index for corner in face.corners], [0, 0, 0])

    def test_rejects_non_triangular_faces(self):
        with self.assertRaisesRegex(ObjParseError, 'triangular'):
            parse_obj_text(
                """
                v 0 0 0
                v 1 0 0
                v 1 1 0
                v 0 1 0
                f 1 2 3 4
                """
            )

    def test_rejects_unsupported_face_token(self):
        with self.assertRaisesRegex(ObjParseError, 'unsupported face token'):
            parse_obj_text(
                """
                v 0 0 0
                v 1 0 0
                v 0 1 0
                vt 0 0
                vt 1 0
                vt 0 1
                vn 0 0 1
                f 1/1/1/1 2/2/1 3/3/1
                """
            )


if __name__ == '__main__':
    unittest.main()
