def export_mesh_to_obj(obj, output_path):
    """Write a deterministic OBJ preserving mesh vertex index to OBJ vertex index mapping.

    Vertices are written in mesh.vertices order and faces are written in
    mesh.polygons order. OBJ indices are 1-based, so Blender vertex index N is
    exported as OBJ vertex N + 1.
    """
    mesh = obj.data
    matrix_world = obj.matrix_world

    for polygon in mesh.polygons:
        if len(polygon.vertices) != 3:
            raise ValueError('OBJ export requires a triangulated mesh.')

    with open(output_path, 'w', encoding='utf-8', newline='\n') as file:
        file.write('# UV Seam Predictor deterministic OBJ\n')
        file.write(f'o {obj.name}\n')

        for vertex in mesh.vertices:
            co = matrix_world @ vertex.co
            file.write(f'v {co.x:.9g} {co.y:.9g} {co.z:.9g}\n')

        for polygon in mesh.polygons:
            indices = [index + 1 for index in polygon.vertices]
            file.write(f'f {indices[0]} {indices[1]} {indices[2]}\n')
