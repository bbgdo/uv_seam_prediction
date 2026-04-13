import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from preprocessing.obj_parser import ObjCorner, ObjFace, ObjMesh, parse_obj
except ModuleNotFoundError:  # pragma: no cover - supports `python preprocessing/topology.py`
    from obj_parser import ObjCorner, ObjFace, ObjMesh, parse_obj


class TopologyError(ValueError):
    pass


EdgeKey = tuple[int, int]


def canonical_edge_key(a: int, b: int) -> EdgeKey:
    if a == b:
        raise TopologyError(f'degenerate edge with repeated vertex id {a}')
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class WeldConfig:
    mode: str = 'exact'
    quantization: float | None = None

    @classmethod
    def exact(cls) -> 'WeldConfig':
        return cls(mode='exact')

    @classmethod
    def welded(cls, quantization: float) -> 'WeldConfig':
        return cls(mode='welded', quantization=quantization)

    def validate(self) -> None:
        if self.mode == 'exact':
            if self.quantization is not None:
                raise TopologyError('exact weld mode does not accept quantization')
            return
        if self.mode == 'welded':
            if self.quantization is None or self.quantization <= 0:
                raise TopologyError('welded mode requires a positive quantization')
            return
        raise TopologyError(f'unsupported weld mode: {self.mode}')


@dataclass(frozen=True)
class FaceEdgeOccurrence:
    face_index: int
    face_line_number: int
    local_edge_index: int
    edge_key: EdgeKey
    corner_a: ObjCorner
    corner_b: ObjCorner


@dataclass(frozen=True)
class CanonicalFace:
    vertex_ids: tuple[int, int, int]
    source_face: ObjFace


@dataclass(frozen=True)
class WeldAudit:
    mode: str
    quantization: float | None
    original_vertex_count: int
    canonical_vertex_count: int
    welded_vertex_count: int
    weld_groups: dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class CanonicalTopology:
    canonical_vertices: tuple[tuple[float, float, float], ...]
    canonical_faces: tuple[CanonicalFace, ...]
    canonical_edges: tuple[EdgeKey, ...]
    edge_incidence: dict[EdgeKey, tuple[FaceEdgeOccurrence, ...]]
    edge_coordinates: dict[EdgeKey, tuple[tuple[float, float, float], tuple[float, float, float]]]
    original_vertex_to_canonical_gid: dict[int, int]
    canonical_gid_to_original_vertex: dict[int, int]
    weld_audit: WeldAudit


def _quantized_key(coords: tuple[float, float, float], quantization: float) -> tuple[int, int, int]:
    return tuple(round(value / quantization) for value in coords)


def _build_vertex_mapping(mesh: ObjMesh, weld_config: WeldConfig) -> tuple[
    list[tuple[float, float, float]],
    dict[int, int],
    dict[int, int],
    WeldAudit,
]:
    weld_config.validate()

    if weld_config.mode == 'exact':
        canonical_vertices = list(mesh.vertices)
        original_to_gid = {idx: idx for idx in range(len(mesh.vertices))}
        gid_to_original = {idx: idx for idx in range(len(mesh.vertices))}
        groups = {idx: (idx,) for idx in range(len(mesh.vertices))}
    else:
        canonical_vertices = []
        original_to_gid = {}
        gid_to_original = {}
        groups_list: dict[int, list[int]] = defaultdict(list)
        bucket_to_gid: dict[tuple[int, int, int], int] = {}

        for original_idx, coords in enumerate(mesh.vertices):
            bucket = _quantized_key(coords, weld_config.quantization)
            if bucket not in bucket_to_gid:
                gid = len(canonical_vertices)
                bucket_to_gid[bucket] = gid
                canonical_vertices.append(coords)
                gid_to_original[gid] = original_idx
            gid = bucket_to_gid[bucket]
            original_to_gid[original_idx] = gid
            groups_list[gid].append(original_idx)
        groups = {gid: tuple(indices) for gid, indices in groups_list.items()}

    welded_vertex_count = sum(max(0, len(indices) - 1) for indices in groups.values())
    audit = WeldAudit(
        mode=weld_config.mode,
        quantization=weld_config.quantization,
        original_vertex_count=len(mesh.vertices),
        canonical_vertex_count=len(canonical_vertices),
        welded_vertex_count=welded_vertex_count,
        weld_groups=groups,
    )
    return canonical_vertices, original_to_gid, gid_to_original, audit


def build_topology(mesh: ObjMesh, weld_config: WeldConfig | None = None) -> CanonicalTopology:
    weld_config = weld_config or WeldConfig.exact()
    canonical_vertices, original_to_gid, gid_to_original, audit = _build_vertex_mapping(mesh, weld_config)

    canonical_faces: list[CanonicalFace] = []
    incidence_lists: dict[EdgeKey, list[FaceEdgeOccurrence]] = defaultdict(list)

    for face_index, face in enumerate(mesh.faces):
        gids = tuple(original_to_gid[corner.vertex_index] for corner in face.corners)
        if len(set(gids)) != 3:
            raise TopologyError(
                f'line {face.line_number}: weld mode produced a degenerate triangle'
            )
        canonical_faces.append(CanonicalFace(vertex_ids=gids, source_face=face))

        for local_edge_index, (a_idx, b_idx) in enumerate(((0, 1), (1, 2), (2, 0))):
            edge_key = canonical_edge_key(gids[a_idx], gids[b_idx])
            incidence_lists[edge_key].append(FaceEdgeOccurrence(
                face_index=face_index,
                face_line_number=face.line_number,
                local_edge_index=local_edge_index,
                edge_key=edge_key,
                corner_a=face.corners[a_idx],
                corner_b=face.corners[b_idx],
            ))

    for edge_key, occurrences in incidence_lists.items():
        if len(occurrences) > 2:
            raise TopologyError(f'non-manifold edge {edge_key}: {len(occurrences)} incident faces')

    canonical_edges = tuple(sorted(incidence_lists))
    edge_incidence = {
        edge_key: tuple(incidence_lists[edge_key])
        for edge_key in canonical_edges
    }
    edge_coordinates = {
        edge_key: (canonical_vertices[edge_key[0]], canonical_vertices[edge_key[1]])
        for edge_key in canonical_edges
    }

    return CanonicalTopology(
        canonical_vertices=tuple(canonical_vertices),
        canonical_faces=tuple(canonical_faces),
        canonical_edges=canonical_edges,
        edge_incidence=edge_incidence,
        edge_coordinates=edge_coordinates,
        original_vertex_to_canonical_gid=original_to_gid,
        canonical_gid_to_original_vertex=gid_to_original,
        weld_audit=audit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Canonical OBJ topology smoke utility.')
    parser.add_argument('obj_path')
    parser.add_argument('--weld-quantization', type=float, default=None)
    args = parser.parse_args()

    mesh = parse_obj(args.obj_path)
    weld_config = (
        WeldConfig.welded(args.weld_quantization)
        if args.weld_quantization is not None
        else WeldConfig.exact()
    )
    topology = build_topology(mesh, weld_config)
    print(f'file: {Path(args.obj_path)}')
    print(f'weld mode: {topology.weld_audit.mode}')
    print(f'canonical vertices: {len(topology.canonical_vertices)}')
    print(f'canonical faces: {len(topology.canonical_faces)}')
    print(f'canonical edges: {len(topology.canonical_edges)}')
    print(f'welded vertices: {topology.weld_audit.welded_vertex_count}')


if __name__ == '__main__':
    main()
