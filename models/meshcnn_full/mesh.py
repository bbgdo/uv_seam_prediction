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


def build_mesh_adjacency(
    faces: np.ndarray,
    unique_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int64)
    if unique_edges is None:
        edge_keys = sorted({
            canonical_edge(int(face[k]), int(face[(k + 1) % 3]))
            for face in faces
            for k in range(3)
        })
        unique_edges = np.asarray(edge_keys, dtype=np.int64).reshape((-1, 2))
    else:
        unique_edges = np.asarray(unique_edges, dtype=np.int64)

    edge_key_to_idx = {
        (int(edge[0]), int(edge[1])): idx
        for idx, edge in enumerate(unique_edges)
    }
    edge_faces: list[list[int]] = [[] for _ in range(len(unique_edges))]
    face_to_edges = np.full((len(faces), 3), -1, dtype=np.int64)

    for face_idx, face in enumerate(faces):
        for local_idx, (a_pos, b_pos) in enumerate(((0, 1), (1, 2), (2, 0))):
            key = canonical_edge(int(face[a_pos]), int(face[b_pos]))
            edge_idx = edge_key_to_idx[key]
            face_to_edges[face_idx, local_idx] = edge_idx
            edge_faces[edge_idx].append(face_idx)

    edge_to_faces = np.full((len(unique_edges), 2), -1, dtype=np.int64)
    for edge_idx, incident in enumerate(edge_faces):
        if len(incident) > 2:
            raise ValueError(f'non-manifold edge {tuple(unique_edges[edge_idx])}: {len(incident)} incident faces')
        for slot, face_idx in enumerate(incident):
            edge_to_faces[edge_idx, slot] = int(face_idx)

    edge_neighbors = np.full((len(unique_edges), 4), -1, dtype=np.int64)
    for edge_idx, incident in enumerate(edge_faces):
        a, b = (int(unique_edges[edge_idx, 0]), int(unique_edges[edge_idx, 1]))
        for face_slot, face_idx in enumerate(incident[:2]):
            face = faces[face_idx]
            opposite = [int(v) for v in face if int(v) not in (a, b)]
            if not opposite:
                continue
            c = opposite[0]
            if face_slot == 0:
                left = canonical_edge(a, c)
                right = canonical_edge(c, b)
                edge_neighbors[edge_idx, 0] = edge_key_to_idx.get(left, -1)
                edge_neighbors[edge_idx, 1] = edge_key_to_idx.get(right, -1)
            else:
                left = canonical_edge(b, c)
                right = canonical_edge(c, a)
                edge_neighbors[edge_idx, 2] = edge_key_to_idx.get(left, -1)
                edge_neighbors[edge_idx, 3] = edge_key_to_idx.get(right, -1)

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
        return cls(
            vertices=sample.vertices.detach().cpu().numpy(),
            faces=sample.faces.detach().cpu().numpy(),
            unique_edges=sample.unique_edges.detach().cpu().numpy(),
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
        new_faces = self.faces.copy()
        new_faces[new_faces == remove] = keep

        kept_faces: list[np.ndarray] = []
        for face_idx, face in enumerate(new_faces):
            if len(set(int(v) for v in face)) != 3:
                if face_idx not in removed_face_set:
                    return 'degenerate triangle result'
                continue
            coords = self.vertices[face]
            area2 = np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
            if not np.isfinite(area2) or area2 <= 1e-12:
                return 'degenerate triangle result'
            kept_faces.append(face)

        if not kept_faces:
            return 'collapse removes all faces'

        edge_counts: dict[tuple[int, int], int] = {}
        face_keys: set[tuple[int, int, int]] = set()
        for face in kept_faces:
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

        new_lookup = {
            (int(edge[0]), int(edge[1])): idx
            for idx, edge in enumerate(self.unique_edges)
        }
        old_to_new = np.full(len(old_edges), -1, dtype=np.int64)
        for old_idx, (u_raw, v_raw) in enumerate(old_edges):
            u = keep if int(u_raw) == remove else int(u_raw)
            v = keep if int(v_raw) == remove else int(v_raw)
            if u == v:
                continue
            key = canonical_edge(u, v)
            old_to_new[old_idx] = new_lookup.get(key, -1)

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
    dataset = torch.load(Path(path), weights_only=False)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f'expected a non-empty list of MeshCNNSample objects, got {type(dataset)}')
    first = dataset[0]
    if not isinstance(first, MeshCNNSample):
        raise ValueError(f'expected MeshCNNSample objects, got {type(first)}')
    return dataset
