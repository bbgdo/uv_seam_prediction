import bmesh


def export_object_to_obj_with_hidden_triangulation(obj, output_path):
    """Export a triangulated in-memory copy while preserving original vertex IDs."""
    mesh = obj.data

    bm = bmesh.new()
    tri_bm = None
    try:
        bm.from_mesh(mesh)
        _ensure_indices(bm)

        vertex_count = len(bm.verts)
        tri_bm = bm.copy()
        _ensure_indices(tri_bm)

        bmesh.ops.triangulate(
            tri_bm,
            faces=list(tri_bm.faces),
            quad_method='FIXED',
            ngon_method='BEAUTY',
        )
        _ensure_indices(tri_bm)

        if len(tri_bm.verts) != vertex_count:
            raise ValueError('Triangulation changed the vertex count; export cancelled.')

        with open(output_path, 'w', encoding='utf-8', newline='\n') as file:
            file.write('# Auto Seams deterministic triangulated OBJ\n')
            file.write(f'o {obj.name}\n')

            for vertex in sorted(bm.verts, key=lambda item: item.index):
                # Keep model input in the mesh's authored coordinate basis. Object
                # location, rotation, and scale are viewport transforms here.
                co = vertex.co
                file.write(f'v {co.x:.9g} {co.y:.9g} {co.z:.9g}\n')

            for face in tri_bm.faces:
                indices = [vertex.index + 1 for vertex in face.verts]
                if len(indices) != 3:
                    raise ValueError('Hidden triangulation produced a non-triangle face.')
                file.write(f'f {indices[0]} {indices[1]} {indices[2]}\n')
    finally:
        if tri_bm is not None:
            tri_bm.free()
        bm.free()


def _ensure_indices(bm):
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
