"""
Blender script for UV unwrapping mesh objects.

Runs inside Blender's Python via:
    blender -b --factory-startup -P evaluation/blender_unwrap.py -- [args]

Three modes (mutually exclusive):
  --seams <file>   mark given edges as seams, then unwrap with ANGLE_BASED/CONFORMAL
  --smart-uv       run Smart UV Project (no manual seams)
  --preserve-uv    re-export the mesh without changing UVs (ground truth passthrough)
"""

import argparse
import math
import os
import sys


def _parse_args() -> argparse.Namespace:
    # Blender passes its own args before '--'; ours start after it
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(description='Blender UV unwrap helper.')
    p.add_argument('--input', required=True, help='Input .obj mesh path')
    p.add_argument('--output', required=True, help='Output .obj path')

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--seams', metavar='FILE',
                      help='Text file with one seam edge index per line '
                           '(0-based into sorted unique edges of the triangulated mesh)')
    mode.add_argument('--smart-uv', action='store_true',
                      help='Use Smart UV Project instead of manual seams')
    mode.add_argument('--preserve-uv', action='store_true',
                      help='Re-export the mesh with its existing UV unchanged')

    p.add_argument('--method', default='ANGLE_BASED', choices=['ANGLE_BASED', 'CONFORMAL'],
                   help='Unwrap algorithm (used with --seams, default: ANGLE_BASED)')
    p.add_argument('--smart-uv-angle', type=float, default=66.0,
                   help='Smart UV Project island angle limit in degrees (default: 66)')

    return p.parse_args(argv)


def _ensure_single_mesh(context) -> object:
    """Return the single mesh object after import, or raise."""
    mesh_objs = [o for o in context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError('No mesh object found after import.')
    return mesh_objs[0]


def _triangulate(obj) -> None:
    """Apply a Triangulate modifier and collapse it."""
    import bpy
    mod = obj.modifiers.new(name='Triangulate', type='TRIANGULATE')
    mod.quad_method = 'BEAUTY'
    mod.ngon_method = 'BEAUTY'
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _sorted_unique_edges(obj) -> list[tuple[int, int]]:
    """Return sorted unique edges (vi < vj) matching our preprocessing convention."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    edges = []
    for e in bm.edges:
        vi = e.verts[0].index
        vj = e.verts[1].index
        edges.append((min(vi, vj), max(vi, vj)))
    bm.free()
    return sorted(edges)


def _export_obj(obj, output_path: str) -> None:
    """Export a single mesh object to .obj preserving UV."""
    import bpy
    # deselect all, select only this object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_selected_objects=True,
        export_uv=True,
        export_normals=True,
        export_materials=False,
        export_triangulated_mesh=True,
    )


def main() -> None:
    import bpy

    args = _parse_args()

    # ── Import ────────────────────────────────────────────────────────────────
    bpy.ops.wm.obj_import(filepath=args.input)
    obj = _ensure_single_mesh(bpy.context)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Switch to object mode to apply modifiers
    bpy.ops.object.mode_set(mode='OBJECT')
    _triangulate(obj)

    # ── Mode: preserve existing UV ────────────────────────────────────────────
    if args.preserve_uv:
        _export_obj(obj, args.output)
        print(f'[blender_unwrap] preserved UV → {args.output}')
        return

    # ── Clear existing UV layers ──────────────────────────────────────────────
    mesh = obj.data
    uv_layers = [uv.name for uv in mesh.uv_layers]
    for name in uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[name])
    mesh.uv_layers.new(name='UVMap')

    # ── Mode: Smart UV Project ────────────────────────────────────────────────
    if args.smart_uv:
        # clear all seams first
        for edge in mesh.edges:
            edge.use_seam = False

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(args.smart_uv_angle))
        bpy.ops.object.mode_set(mode='OBJECT')

        _export_obj(obj, args.output)
        print(f'[blender_unwrap] smart UV project → {args.output}')
        return

    # ── Mode: Unwrap with given seam indices ──────────────────────────────────
    with open(args.seams) as sf:
        seam_set = {int(line.strip()) for line in sf if line.strip()}

    # clear all seams
    for edge in mesh.edges:
        edge.use_seam = False

    unique_edges = _sorted_unique_edges(obj)

    # build lookup: (vi, vj) -> mesh edge index
    edge_key_to_mesh_idx = {
        (min(e.vertices[0], e.vertices[1]),
         max(e.vertices[0], e.vertices[1])): i
        for i, e in enumerate(mesh.edges)
    }

    marked = 0
    for seam_idx in seam_set:
        if seam_idx >= len(unique_edges):
            continue
        vi, vj = unique_edges[seam_idx]
        key = (min(vi, vj), max(vi, vj))
        mesh_idx = edge_key_to_mesh_idx.get(key)
        if mesh_idx is not None:
            mesh.edges[mesh_idx].use_seam = True
            marked += 1

    print(f'[blender_unwrap] marked {marked} / {len(seam_set)} seam edges')

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method=args.method, margin=0.001)
    bpy.ops.object.mode_set(mode='OBJECT')

    _export_obj(obj, args.output)
    print(f'[blender_unwrap] unwrap ({args.method}) → {args.output}')


if __name__ == '__main__':
    main()
