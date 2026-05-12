from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

from models.meshcnn_full.mesh import MeshCNNSample, build_mesh_adjacency
from models.utils.seam_topology import build_seam_graph_view, compute_seam_mask_diagnostics, diagnostics_to_json_dict
from preprocessing.build_gnn_dataset import build_dual_edge_index_from_unique_edges
from preprocessing.feature_registry import ResolvedFeatureSet
from preprocessing.label_sources import INFERENCE_LABEL_SOURCE
from preprocessing.topology import CanonicalTopology
from tools.utils.prediction_common import PredictionError


def build_feature_mesh_from_canonical_topology(topology: CanonicalTopology) -> trimesh.Trimesh:
    vertices = np.asarray(topology.canonical_vertices, dtype=np.float64)
    faces = np.asarray([face.vertex_ids for face in topology.canonical_faces], dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise PredictionError('input OBJ produced an empty feature mesh', 'InvalidMesh')
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def assert_canonical_edge_order(unique_edges: np.ndarray, canonical_edges: tuple, mesh_path: Path) -> None:
    expected_edges = np.asarray(canonical_edges, dtype=np.int64).reshape((-1, 2))
    if np.array_equal(unique_edges, expected_edges):
        return

    if unique_edges.shape != expected_edges.shape:
        detail = f'feature_edges shape={unique_edges.shape}, canonical_edges shape={expected_edges.shape}'
    else:
        mismatch_rows = np.flatnonzero(np.any(unique_edges != expected_edges, axis=1))
        idx = int(mismatch_rows[0]) if len(mismatch_rows) else -1
        detail = f'first mismatch at {idx}: feature={tuple(unique_edges[idx])}, canonical={tuple(expected_edges[idx])}'
    raise PredictionError(
        f'canonical edge order mismatch for {mesh_path}: {detail}',
        'EdgeOrderMismatch',
    )


def build_dual_data(edge_features: np.ndarray, unique_edges: np.ndarray) -> Data:
    dual_edge_index = build_dual_edge_index_from_unique_edges(unique_edges)
    dual_x = torch.from_numpy(edge_features).float()
    return Data(x=dual_x, edge_index=dual_edge_index, num_nodes=len(unique_edges))


def build_meshcnn_inference_sample(
    *,
    mesh_path: Path,
    feature_mesh: trimesh.Trimesh,
    unique_edges: np.ndarray,
    edge_features: np.ndarray,
    selection: ResolvedFeatureSet,
    endpoint_order: str,
    topology: CanonicalTopology,
) -> MeshCNNSample:
    faces = np.asarray(feature_mesh.faces, dtype=np.int64)
    unique_edges, edge_to_faces, face_to_edges, edge_neighbors, boundary_mask = build_mesh_adjacency(
        faces,
        np.asarray(unique_edges, dtype=np.int64),
    )
    return MeshCNNSample(
        vertices=torch.from_numpy(np.asarray(feature_mesh.vertices, dtype=np.float32)),
        faces=torch.from_numpy(faces.astype(np.int64, copy=False)),
        unique_edges=torch.from_numpy(unique_edges.astype(np.int64, copy=False)),
        edge_features=torch.from_numpy(np.asarray(edge_features, dtype=np.float32)),
        edge_labels=torch.zeros(len(unique_edges), dtype=torch.float32),
        edge_neighbors=torch.from_numpy(edge_neighbors.astype(np.int64, copy=False)),
        edge_to_faces=torch.from_numpy(edge_to_faces.astype(np.int64, copy=False)),
        face_to_edges=torch.from_numpy(face_to_edges.astype(np.int64, copy=False)),
        boundary_mask=torch.from_numpy(boundary_mask.astype(bool, copy=False)),
        file_path=str(mesh_path),
        feature_group=selection.feature_group,
        feature_names=list(selection.feature_names),
        feature_flags=selection.feature_flags.as_dict(),
        density_config=dict(selection.density_config) if selection.density_config else None,
        endpoint_order=endpoint_order,
        label_source=INFERENCE_LABEL_SOURCE,
        weld_mode=topology.weld_audit.mode,
        seam_edge_count=0,
        boundary_edge_count=int(np.count_nonzero(boundary_mask)),
    )


def json_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def json_vector(values: np.ndarray) -> list[float | None]:
    return [json_float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def build_mesh_diagnostics(
    feature_mesh: trimesh.Trimesh,
    edge_features: np.ndarray,
    probabilities: np.ndarray | None = None,
    threshold: float | None = None,
    topology: CanonicalTopology | None = None,
    unique_edges: np.ndarray | None = None,
) -> dict[str, Any]:
    vertices = np.asarray(feature_mesh.vertices, dtype=np.float64)
    faces = np.asarray(feature_mesh.faces, dtype=np.int64)
    if len(vertices):
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        centroid = vertices.mean(axis=0)
    else:
        bounds_min = np.zeros(3, dtype=np.float64)
        bounds_max = np.zeros(3, dtype=np.float64)
        centroid = np.zeros(3, dtype=np.float64)
    size = bounds_max - bounds_min
    diag = float(np.linalg.norm(size))
    features = np.asarray(edge_features, dtype=np.float64)
    finite_features = features[np.isfinite(features)]
    diagnostics = {
        'coordinate_space': {
            'exported_basis': 'mesh_local',
            'object_matrix_applied': False,
            'transform': 'p_export = I * p_mesh_local',
        },
        'mesh_bbox': {
            'vertex_count': int(len(vertices)),
            'face_count': int(len(faces)),
            'min': json_vector(bounds_min),
            'max': json_vector(bounds_max),
            'size': json_vector(size),
            'diagonal': json_float(diag),
            'centroid': json_vector(centroid),
            'finite_vertices': bool(np.isfinite(vertices).all()) if len(vertices) else True,
        },
        'edge_features': {
            'shape': [int(dim) for dim in features.shape],
            'finite': bool(np.isfinite(features).all()) if features.size else True,
            'min': json_float(finite_features.min()) if finite_features.size else None,
            'max': json_float(finite_features.max()) if finite_features.size else None,
            'mean': json_float(finite_features.mean()) if finite_features.size else None,
        },
    }
    if (
        probabilities is not None
        and threshold is not None
        and topology is not None
        and unique_edges is not None
    ):
        seam_graph_view = build_seam_graph_view(
            topology=topology,
            unique_edges=np.asarray(unique_edges, dtype=np.int64),
        )
        diagnostics['seam_topology'] = diagnostics_to_json_dict(
            compute_seam_mask_diagnostics(
                view=seam_graph_view,
                probabilities=np.asarray(probabilities, dtype=np.float64),
                threshold=float(threshold),
            )
        )
    else:
        diagnostics['seam_topology'] = None
    return diagnostics
