"""Extract body mesh from Autodesk Character Generator FBX exports.

Finds the 'H_DDS_MidRes' mesh, unparents from rig (keeping transforms),
and exports as OBJ with UVs preserved.

Usage:
    blender --background --python preprocessing/autodesk_char_gen/fbx_to_obj.py -- <fbx_dir> [--out <output_dir>]
"""

import os
import sys

import bpy


MESH_NAME_PATTERN = 'H_DDS_MidRes'


def reset_scene():
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for pool in (bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                 bpy.data.images, bpy.data.armatures, bpy.data.libraries,
                 bpy.data.lights, bpy.data.cameras):
        for item in pool:
            try:
                pool.remove(item)
            except Exception:
                pass


def find_body_mesh():
    """Find the H_DDS_MidRes mesh object in the scene."""
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and MESH_NAME_PATTERN in obj.name:
            return obj
    return None


def unparent_keep_transform(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')


def export_obj(obj, out_path):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        bpy.ops.wm.obj_export(
            filepath=out_path,
            export_selected_objects=True,
            apply_modifiers=True,
            export_uv=True,
            export_normals=True,
            export_triangulated_mesh=True,
            export_materials=False,
            forward_axis='NEGATIVE_Z',
            up_axis='Y',
        )
    except AttributeError:
        bpy.ops.export_scene.obj(
            filepath=out_path,
            use_selection=True,
            use_mesh_modifiers=True,
            use_normals=True,
            use_uvs=True,
            use_triangles=False,
            use_materials=False,
            axis_forward='-Z',
            axis_up='Y',
        )


def process_fbx(fbx_path, output_dir):
    name = os.path.splitext(os.path.basename(fbx_path))[0]

    reset_scene()
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    body = find_body_mesh()
    if body is None:
        print(f'  [skip] {name}: no "{MESH_NAME_PATTERN}" mesh found')
        return False

    unparent_keep_transform(body)

    out_path = os.path.join(output_dir, f'{name}.obj')
    export_obj(body, out_path)
    print(f'  [ok] {name} -> {os.path.basename(out_path)}')
    return True


def main():
    argv = sys.argv
    if '--' not in argv:
        print('Usage: blender --background --python fbx_to_obj.py -- <fbx_dir> [--out <output_dir>]')
        return

    args = argv[argv.index('--') + 1:]
    if not args:
        print('Error: no input directory specified')
        return

    fbx_dir = os.path.abspath(args[0])
    output_dir = fbx_dir

    if '--out' in args:
        out_idx = args.index('--out')
        if out_idx + 1 < len(args):
            output_dir = os.path.abspath(args[out_idx + 1])

    if not os.path.isdir(fbx_dir):
        print(f'Error: directory not found: {fbx_dir}')
        return

    os.makedirs(output_dir, exist_ok=True)

    fbx_files = sorted(f for f in os.listdir(fbx_dir) if f.lower().endswith('.fbx'))
    if not fbx_files:
        print(f'No .fbx files found in {fbx_dir}')
        return

    print(f'\nProcessing {len(fbx_files)} FBX file(s)')
    print(f'Input:  {fbx_dir}')
    print(f'Output: {output_dir}\n')

    ok, fail = 0, 0
    for f in fbx_files:
        if process_fbx(os.path.join(fbx_dir, f), output_dir):
            ok += 1
        else:
            fail += 1

    print(f'\nDone. Exported: {ok}, Skipped: {fail}')


if __name__ == '__main__':
    main()