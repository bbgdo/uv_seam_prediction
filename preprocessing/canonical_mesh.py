import numpy as np
import trimesh


def resolve_endpoint_order(feature_group: str, endpoint_order: str) -> str:
    if endpoint_order != 'auto':
        return endpoint_order
    return 'random' if feature_group == 'paper14' else 'fixed'


def build_feature_mesh_from_topology(
    topology,
    empty_message: str = 'exact_obj requires a non-empty OBJ mesh',
) -> trimesh.Trimesh:
    vertices = np.asarray(topology.canonical_vertices, dtype=np.float64)
    faces = np.asarray([face.vertex_ids for face in topology.canonical_faces], dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(empty_message)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
