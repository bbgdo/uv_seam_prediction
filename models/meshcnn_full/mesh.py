from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from preprocessing.label_sources import EXACT_OBJ_LABEL_SOURCE


SPARSE_MESHCNN_SAMPLE_FORMAT = 'sparsemeshcnn_v2'


@dataclass
class MeshCNNSample:
    vertices: torch.Tensor
    faces: torch.Tensor
    unique_edges: torch.Tensor
    edge_features: torch.Tensor
    edge_labels: torch.Tensor
    edge_neighbors: torch.Tensor
    edge_to_faces: torch.Tensor
    face_to_edges: torch.Tensor
    boundary_mask: torch.Tensor
    file_path: str
    feature_group: str
    feature_names: list[str]
    feature_flags: dict[str, bool]
    endpoint_order: str
    label_source: str = EXACT_OBJ_LABEL_SOURCE
    weld_mode: str = 'exact'
    density_config: dict[str, Any] | None = None
    seam_edge_count: int = 0
    boundary_edge_count: int = 0
    sparse_cache: dict[str, Any] | None = None

    @property
    def y(self) -> torch.Tensor:
        return self.edge_labels

    @property
    def x(self) -> torch.Tensor:
        return self.edge_features

    @property
    def num_edges(self) -> int:
        return int(self.edge_features.shape[0])

    @property
    def in_channels(self) -> int:
        return int(self.edge_features.shape[1])

    def to(self, device: torch.device | str) -> 'MeshCNNSample':
        tensor_names = (
            'vertices',
            'faces',
            'unique_edges',
            'edge_features',
            'edge_labels',
            'edge_neighbors',
            'edge_to_faces',
            'face_to_edges',
            'boundary_mask',
        )
        values = self.__dict__.copy()
        for name in tensor_names:
            values[name] = values[name].to(device)
        return MeshCNNSample(**_sample_constructor_values(values))


def _sample_field_names() -> frozenset[str]:
    return frozenset(field.name for field in fields(MeshCNNSample))


def _sample_constructor_values(values: dict[str, Any]) -> dict[str, Any]:
    field_names = _sample_field_names()
    return {key: value for key, value in values.items() if key in field_names}


def _sample_tensor_names() -> tuple[str, ...]:
    return (
        'vertices',
        'faces',
        'unique_edges',
        'edge_features',
        'edge_labels',
        'edge_neighbors',
        'edge_to_faces',
        'face_to_edges',
        'boundary_mask',
    )


def _ensure_sample_cpu(sample: MeshCNNSample) -> MeshCNNSample:
    values = sample.__dict__.copy()
    for name in _sample_tensor_names():
        tensor = values[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{name} is not a tensor: {type(tensor)}')
        values[name] = tensor.detach().to(device='cpu', copy=False).contiguous()
    return MeshCNNSample(**_sample_constructor_values(values))


def _edge_search_indices(unique_edges: np.ndarray, query_edges: np.ndarray) -> np.ndarray:
    unique_edges = np.ascontiguousarray(unique_edges, dtype=np.int64)
    query_edges = np.ascontiguousarray(query_edges, dtype=np.int64)
    if len(unique_edges) > 1:
        sorted_order = (
            (unique_edges[:-1, 0] < unique_edges[1:, 0])
            | (
                (unique_edges[:-1, 0] == unique_edges[1:, 0])
                & (unique_edges[:-1, 1] <= unique_edges[1:, 1])
            )
        )
        if not bool(np.all(sorted_order)):
            lookup = {
                (int(edge[0]), int(edge[1])): idx
                for idx, edge in enumerate(unique_edges)
            }
            return np.asarray(
                [lookup.get((int(edge[0]), int(edge[1])), -1) for edge in query_edges],
                dtype=np.int64,
            )
    key_dtype = np.dtype([('u', np.int64), ('v', np.int64)])
    unique_keys = unique_edges.view(key_dtype).reshape(-1)
    query_keys = query_edges.view(key_dtype).reshape(-1)
    positions = np.searchsorted(unique_keys, query_keys)
    found = positions < len(unique_keys)
    safe_positions = np.minimum(positions, max(len(unique_keys) - 1, 0))
    if len(unique_keys):
        found &= unique_keys[safe_positions] == query_keys
    out = np.full(len(query_edges), -1, dtype=np.int64)
    out[found] = positions[found]
    return out


def build_mesh_adjacency(
    faces: np.ndarray,
    unique_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int64)
    if unique_edges is None:
        raw_edges = np.stack(
            (
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ),
            axis=1,
        ).reshape((-1, 2))
        edge_pairs = np.sort(raw_edges, axis=1)
        degenerate = edge_pairs[:, 0] == edge_pairs[:, 1]
        if np.any(degenerate):
            vertex = int(edge_pairs[degenerate][0, 0])
            raise ValueError(f'degenerate edge with repeated vertex id {vertex}')
        unique_edges, inverse = np.unique(edge_pairs, axis=0, return_inverse=True)
        face_to_edges = inverse.reshape((len(faces), 3)).astype(np.int64, copy=False)
    else:
        unique_edges = np.asarray(unique_edges, dtype=np.int64)
        raw_edges = np.stack(
            (
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ),
            axis=1,
        ).reshape((-1, 2))
        edge_pairs = np.sort(raw_edges, axis=1)
        degenerate = edge_pairs[:, 0] == edge_pairs[:, 1]
        if np.any(degenerate):
            vertex = int(edge_pairs[degenerate][0, 0])
            raise ValueError(f'degenerate edge with repeated vertex id {vertex}')
        face_to_edges = _edge_search_indices(unique_edges, edge_pairs).reshape((len(faces), 3))
        if np.any(face_to_edges < 0):
            missing = edge_pairs[face_to_edges.reshape(-1) < 0][0]
            raise KeyError((int(missing[0]), int(missing[1])))

    edge_to_faces = np.full((len(unique_edges), 2), -1, dtype=np.int64)
    occurrence_edges = face_to_edges.reshape(-1)
    occurrence_faces = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    incident_counts = np.bincount(occurrence_edges, minlength=len(unique_edges))
    nonmanifold = np.flatnonzero(incident_counts > 2)
    if len(nonmanifold):
        edge_idx = int(nonmanifold[0])
        raise ValueError(f'non-manifold edge {tuple(unique_edges[edge_idx])}: {int(incident_counts[edge_idx])} incident faces')

    order = np.argsort(occurrence_edges, kind='stable')
    sorted_edges = occurrence_edges[order]
    sorted_faces = occurrence_faces[order]
    first = np.ones(len(sorted_edges), dtype=bool)
    first[1:] = sorted_edges[1:] != sorted_edges[:-1]
    second = ~first
    edge_to_faces[sorted_edges[first], 0] = sorted_faces[first]
    edge_to_faces[sorted_edges[second], 1] = sorted_faces[second]

    edge_neighbors = np.full((len(unique_edges), 4), -1, dtype=np.int64)
    if len(faces):
        local_edges = face_to_edges
        edge_ids = local_edges.reshape(-1)
        left_neighbors = np.stack(
            (
                local_edges[:, 2],
                local_edges[:, 0],
                local_edges[:, 1],
            ),
            axis=1,
        ).reshape(-1)
        right_neighbors = np.stack(
            (
                local_edges[:, 1],
                local_edges[:, 2],
                local_edges[:, 0],
            ),
            axis=1,
        ).reshape(-1)
        face_ids = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
        base_slots = np.where(edge_to_faces[edge_ids, 0] == face_ids, 0, 2)
        edge_neighbors[edge_ids, base_slots] = left_neighbors
        edge_neighbors[edge_ids, base_slots + 1] = right_neighbors

    boundary_mask = edge_to_faces[:, 1] < 0
    return unique_edges, edge_to_faces, face_to_edges, edge_neighbors, boundary_mask


def load_meshcnn_dataset(path: str | Path) -> list[MeshCNNSample]:
    dataset = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f'expected a non-empty list of MeshCNNSample objects, got {type(dataset)}')
    for sample in dataset:
        if not isinstance(sample, MeshCNNSample):
            raise ValueError(f'expected MeshCNNSample objects, got {type(sample)}')
    return [_ensure_sample_cpu(sample) for sample in dataset]
