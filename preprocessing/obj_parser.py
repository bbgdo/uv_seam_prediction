import argparse
from dataclasses import dataclass
from pathlib import Path


class ObjParseError(ValueError):
    pass


@dataclass(frozen=True)
class ObjCorner:
    vertex_index: int
    uv_index: int | None = None
    normal_index: int | None = None


@dataclass(frozen=True)
class ObjFace:
    corners: tuple[ObjCorner, ObjCorner, ObjCorner]
    line_number: int


@dataclass(frozen=True)
class ObjMesh:
    vertices: tuple[tuple[float, float, float], ...]
    uvs: tuple[tuple[float, float], ...]
    normals: tuple[tuple[float, float, float], ...]
    faces: tuple[ObjFace, ...]
    file_path: str | None = None


def _parse_float_triplet(parts: list[str], line_number: int, kind: str) -> tuple[float, float, float]:
    if len(parts) < 4:
        raise ObjParseError(f'line {line_number}: {kind} requires three coordinates')
    try:
        return float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError as exc:
        raise ObjParseError(f'line {line_number}: invalid {kind} coordinate') from exc


def _parse_uv(parts: list[str], line_number: int) -> tuple[float, float]:
    if len(parts) < 3:
        raise ObjParseError(f'line {line_number}: vt requires at least two coordinates')
    try:
        return float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise ObjParseError(f'line {line_number}: invalid vt coordinate') from exc


def _resolve_obj_index(raw: str, count: int, line_number: int, label: str) -> int:
    if raw == '':
        raise ObjParseError(f'line {line_number}: missing {label} index')
    try:
        value = int(raw)
    except ValueError as exc:
        raise ObjParseError(f'line {line_number}: invalid {label} index {raw!r}') from exc

    if value == 0:
        raise ObjParseError(f'line {line_number}: OBJ indices are 1-based; got 0')
    index = value - 1 if value > 0 else count + value
    if index < 0 or index >= count:
        raise ObjParseError(f'line {line_number}: {label} index {value} is out of range')
    return index


def _parse_face_token(
    token: str,
    vertex_count: int,
    uv_count: int,
    normal_count: int,
    line_number: int,
) -> ObjCorner:
    pieces = token.split('/')
    if len(pieces) == 1:
        vertex_index = _resolve_obj_index(pieces[0], vertex_count, line_number, 'vertex')
        return ObjCorner(vertex_index=vertex_index)
    if len(pieces) == 2:
        vertex_index = _resolve_obj_index(pieces[0], vertex_count, line_number, 'vertex')
        uv_index = _resolve_obj_index(pieces[1], uv_count, line_number, 'uv')
        return ObjCorner(vertex_index=vertex_index, uv_index=uv_index)
    if len(pieces) == 3:
        vertex_index = _resolve_obj_index(pieces[0], vertex_count, line_number, 'vertex')
        if pieces[1] == '':
            normal_index = _resolve_obj_index(pieces[2], normal_count, line_number, 'normal')
            return ObjCorner(vertex_index=vertex_index, normal_index=normal_index)
        uv_index = _resolve_obj_index(pieces[1], uv_count, line_number, 'uv')
        normal_index = _resolve_obj_index(pieces[2], normal_count, line_number, 'normal')
        return ObjCorner(vertex_index=vertex_index, uv_index=uv_index, normal_index=normal_index)
    raise ObjParseError(f'line {line_number}: unsupported face token {token!r}')


def parse_obj_text(text: str, file_path: str | Path | None = None) -> ObjMesh:
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    faces: list[ObjFace] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue

        parts = line.split()
        kind = parts[0]
        if kind == 'v':
            vertices.append(_parse_float_triplet(parts, line_number, 'v'))
        elif kind == 'vt':
            uvs.append(_parse_uv(parts, line_number))
        elif kind == 'vn':
            normals.append(_parse_float_triplet(parts, line_number, 'vn'))
        elif kind == 'f':
            if len(parts) != 4:
                raise ObjParseError(f'line {line_number}: only triangular faces are supported')
            corners = tuple(
                _parse_face_token(token, len(vertices), len(uvs), len(normals), line_number)
                for token in parts[1:]
            )
            faces.append(ObjFace(corners=corners, line_number=line_number))

    return ObjMesh(
        vertices=tuple(vertices),
        uvs=tuple(uvs),
        normals=tuple(normals),
        faces=tuple(faces),
        file_path=str(file_path) if file_path is not None else None,
    )


def parse_obj(path: str | Path) -> ObjMesh:
    obj_path = Path(path)
    text = obj_path.read_text(encoding='utf-8', errors='replace')
    return parse_obj_text(text, file_path=obj_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Strict OBJ parser smoke utility.')
    parser.add_argument('obj_path')
    args = parser.parse_args()

    mesh = parse_obj(args.obj_path)
    print(f'file: {mesh.file_path}')
    print(f'vertices: {len(mesh.vertices)}')
    print(f'uvs: {len(mesh.uvs)}')
    print(f'normals: {len(mesh.normals)}')
    print(f'faces: {len(mesh.faces)}')


if __name__ == '__main__':
    main()
