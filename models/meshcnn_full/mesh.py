from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
    feature_preset: str
    feature_names: list[str]
    feature_flags: dict[str, bool]
    endpoint_order: str
    label_source: str = 'exact_obj'
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
        return MeshCNNSample(**values)


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
    return MeshCNNSample(**values)


@dataclass
class CollapseHistory:
    old_edges: torch.Tensor
    new_edges: torch.Tensor
    old_to_new: torch.Tensor
    collapsed_edges: list[tuple[int, int]]
    old_edge_count: int
    new_edge_count: int


@dataclass
class EdgeCollapseRecord:
    edge_key: tuple[int, int]
    kept_vertex: int
    removed_vertex: int
    removed_faces: list[int]
    old_to_new: np.ndarray
    old_edges: np.ndarray
    new_edges: np.ndarray


def canonical_edge(a: int, b: int) -> tuple[int, int]:
    if a == b:
        raise ValueError(f'degenerate edge with repeated vertex id {a}')
    return (a, b) if a < b else (b, a)


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


class MutableMeshTopology:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        unique_edges: np.ndarray | None = None,
    ):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int64).copy()
        self._rebuild(unique_edges)

    @classmethod
    def from_sample(cls, sample: MeshCNNSample) -> 'MutableMeshTopology':
        tensors = {
            'vertices': sample.vertices,
            'faces': sample.faces,
            'unique_edges': sample.unique_edges,
        }
        for name, tensor in tensors.items():
            if tensor.device.type != 'cpu':
                raise RuntimeError(
                    f'MeshCNNSample.{name} must be on CPU before topology reconstruction, '
                    f'got {tensor.device}. Dataset loading must normalize samples to CPU.'
                )
        return cls(
            vertices=sample.vertices.detach().numpy(),
            faces=sample.faces.detach().numpy(),
            unique_edges=sample.unique_edges.detach().numpy(),
        )

    def clone(self) -> 'MutableMeshTopology':
        return MutableMeshTopology(self.vertices.copy(), self.faces.copy(), self.unique_edges.copy())

    @property
    def edge_count(self) -> int:
        return int(len(self.unique_edges))

    def _rebuild(self, unique_edges: np.ndarray | None = None) -> None:
        (
            self.unique_edges,
            self.edge_to_faces,
            self.face_to_edges,
            self.edge_neighbors,
            self.boundary_mask,
        ) = build_mesh_adjacency(self.faces, unique_edges)
        self.edge_key_to_idx = {
            (int(edge[0]), int(edge[1])): idx
            for idx, edge in enumerate(self.unique_edges)
        }
        self.vertex_to_faces: dict[int, set[int]] = {}
        for face_idx, face in enumerate(self.faces):
            for vertex in face:
                self.vertex_to_faces.setdefault(int(vertex), set()).add(int(face_idx))

    def neighbors_tensor(self, device: torch.device | str) -> torch.Tensor:
        return torch.as_tensor(self.edge_neighbors, dtype=torch.long, device=device)

    def is_valid_collapse(self, edge_idx: int) -> bool:
        return self.collapse_error(edge_idx) is None

    def collapse_error(self, edge_idx: int) -> str | None:
        if edge_idx < 0 or edge_idx >= self.edge_count:
            return 'edge index out of range'
        if bool(self.boundary_mask[edge_idx]):
            return 'boundary edge'

        incident = [int(f) for f in self.edge_to_faces[edge_idx] if int(f) >= 0]
        if len(incident) != 2:
            return 'collapse requires exactly two incident faces'

        a, b = (int(self.unique_edges[edge_idx, 0]), int(self.unique_edges[edge_idx, 1]))
        keep, remove = (a, b) if a < b else (b, a)
        removed_face_set = set(incident)

        # Guard collapse validation to the finite local star affected by the merge.
        affected_faces = set(self.vertex_to_faces.get(keep, ()))
        affected_faces.update(self.vertex_to_faces.get(remove, ()))
        if len(affected_faces) > len(self.faces) or not removed_face_set.issubset(affected_faces):
            return 'inconsistent topology'

        kept_faces: list[tuple[int, np.ndarray]] = []
        for face_idx in affected_faces:
            if face_idx < 0 or face_idx >= len(self.faces):
                return 'inconsistent topology'
            face = self.faces[face_idx].copy()
            face[face == remove] = keep
            if len(set(int(v) for v in face)) != 3:
                if face_idx not in removed_face_set:
                    return 'degenerate triangle result'
                continue
            coords = self.vertices[face]
            area2 = np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
            if not np.isfinite(area2) or area2 <= 1e-12:
                return 'degenerate triangle result'
            kept_faces.append((face_idx, face))

        if len(self.faces) - len(affected_faces) + len(kept_faces) == 0:
            return 'collapse removes all faces'

        edge_counts: dict[tuple[int, int], int] = {}
        face_keys: set[tuple[int, int, int]] = set()
        for _, face in kept_faces:
            face_key = tuple(sorted(int(v) for v in face))
            if face_key in face_keys:
                return 'duplicate triangle result'
            face_keys.add(face_key)
            for k in range(3):
                key = canonical_edge(int(face[k]), int(face[(k + 1) % 3]))
                edge_counts[key] = edge_counts.get(key, 0) + 1
                if edge_counts[key] > 2:
                    return 'non-manifold result'

        return None

    def collapse_edge(self, edge_idx: int) -> EdgeCollapseRecord:
        error = self.collapse_error(edge_idx)
        if error is not None:
            raise ValueError(f'invalid collapse for edge {edge_idx}: {error}')

        old_edges = self.unique_edges.copy()
        edge_key = (int(old_edges[edge_idx, 0]), int(old_edges[edge_idx, 1]))
        keep, remove = edge_key if edge_key[0] < edge_key[1] else (edge_key[1], edge_key[0])
        removed_faces = [int(f) for f in self.edge_to_faces[edge_idx] if int(f) >= 0]

        new_faces = self.faces.copy()
        new_faces[new_faces == remove] = keep
        keep_mask = np.array([len(set(int(v) for v in face)) == 3 for face in new_faces], dtype=bool)
        self.faces = new_faces[keep_mask]
        self._rebuild()

        old_to_new = np.full(len(old_edges), -1, dtype=np.int64)
        remapped_edges = old_edges.copy()
        remapped_edges[remapped_edges == remove] = keep
        valid = remapped_edges[:, 0] != remapped_edges[:, 1]
        if np.any(valid):
            query_edges = np.sort(remapped_edges[valid], axis=1)
            old_to_new[valid] = _edge_search_indices(self.unique_edges, query_edges)

        return EdgeCollapseRecord(
            edge_key=edge_key,
            kept_vertex=keep,
            removed_vertex=remove,
            removed_faces=removed_faces,
            old_to_new=old_to_new,
            old_edges=old_edges,
            new_edges=self.unique_edges.copy(),
        )


def make_collapse_history(
    records: list[EdgeCollapseRecord],
    old_edges: np.ndarray,
    new_edges: np.ndarray,
    device: torch.device | str,
) -> CollapseHistory:
    if records:
        old_to_new = records[-1].old_to_new
        collapsed = [record.edge_key for record in records]
    else:
        old_to_new = np.arange(len(old_edges), dtype=np.int64)
        collapsed = []
    return CollapseHistory(
        old_edges=torch.as_tensor(old_edges, dtype=torch.long, device=device),
        new_edges=torch.as_tensor(new_edges, dtype=torch.long, device=device),
        old_to_new=torch.as_tensor(old_to_new, dtype=torch.long, device=device),
        collapsed_edges=collapsed,
        old_edge_count=int(len(old_edges)),
        new_edge_count=int(len(new_edges)),
    )


def load_meshcnn_dataset(path: str | Path) -> list[MeshCNNSample]:
    dataset = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f'expected a non-empty list of MeshCNNSample objects, got {type(dataset)}')
    for sample in dataset:
        if not isinstance(sample, MeshCNNSample):
            raise ValueError(f'expected MeshCNNSample objects, got {type(sample)}')
    return [_ensure_sample_cpu(sample) for sample in dataset]
