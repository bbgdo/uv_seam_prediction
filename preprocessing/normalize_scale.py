import bpy
import os
import sys
import mathutils

EXPORT_CONFIG = {
    'use_selection': True,
    'use_mesh_modifiers': True,
    'use_normals': True,
    'use_uvs': True,
    'use_triangles': False,
    'use_materials': False,
    'axis_forward': '-Z',
    'axis_up': 'Y',
}


def reset_scene():
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images, bpy.data.libraries]:
        for item in collection:
            try:
                collection.remove(item)
            except:
                pass


def get_unique_filepath(directory, filename):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1
    return os.path.join(directory, new_filename)


def normalize_objects():
    selected_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not selected_objects:
        return False

    min_v = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_v = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))

    vertices_found = False

    for obj in selected_objects:
        mw = obj.matrix_world
        bbox = [mw @ mathutils.Vector(corner) for corner in obj.bound_box]

        for v in bbox:
            min_v.x = min(min_v.x, v.x)
            min_v.y = min(min_v.y, v.y)
            min_v.z = min(min_v.z, v.z)

            max_v.x = max(max_v.x, v.x)
            max_v.y = max(max_v.y, v.y)
            max_v.z = max(max_v.z, v.z)
            vertices_found = True

    if not vertices_found:
        return False

    center = (min_v + max_v) / 2
    dimensions = max_v - min_v
    diagonal = dimensions.length

    if diagonal == 0:
        return False

    # center + scale to unit diagonal
    for obj in selected_objects:
        obj.location -= center

    scale_factor = 1.0 / diagonal
    for obj in selected_objects:
        obj.scale *= scale_factor

    # bake transforms so .obj has raw coords in [-0.5, 0.5]
    bpy.context.view_layer.objects.active = selected_objects[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    return True


def export_obj_modern(out_path):
    try:
        bpy.ops.wm.obj_export(
            filepath=out_path,
            export_selected_objects=True,
            apply_modifiers=EXPORT_CONFIG['use_mesh_modifiers'],
            export_uv=EXPORT_CONFIG['use_uvs'],
            export_normals=EXPORT_CONFIG['use_normals'],
            export_triangulated_mesh=EXPORT_CONFIG['use_triangles'],
            export_materials=False,
            forward_axis='NEGATIVE_Z',
            up_axis='Y'
        )
    except AttributeError:
        bpy.ops.export_scene.obj(
            filepath=out_path,
            **EXPORT_CONFIG
        )


def process_directory(input_path_arg):
    input_dir = os.path.abspath(input_path_arg)
    if not os.path.exists(input_dir):
        print(f"Error: Dir '{input_dir}' is not found.")
        return

    parent_dir = os.path.dirname(input_dir.rstrip(os.sep))
    base_name = os.path.basename(input_dir.rstrip(os.sep))
    output_dir = os.path.join(parent_dir, f"{base_name}_normalized")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.obj')]

    print(f"\n=== Normalization Pipeline ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files: {len(files)}")
    print("==============================\n")

    success_count = 0
    fail_count = 0

    for f in files:
        full_path = os.path.join(input_dir, f)

        sys.stdout.write(f"Processing: {f} ... ")
        sys.stdout.flush()

        reset_scene()

        try:
            if hasattr(bpy.ops.wm, "obj_import"):
                bpy.ops.wm.obj_import(filepath=full_path)
            else:
                bpy.ops.import_scene.obj(filepath=full_path)
        except Exception as e:
            print(f"FAIL (Import): {e}")
            fail_count += 1
            continue

        if normalize_objects():
            target_file = f
            out_path = get_unique_filepath(output_dir, target_file)

            try:
                export_obj_modern(out_path)
                print("OK")
                success_count += 1
            except Exception as e:
                print(f"FAIL (Export): {e}")
                fail_count += 1
        else:
            print("FAIL (Empty/Error)")
            fail_count += 1

    print(f"\n=== Done ===")
    print(f"Processed: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
        if len(argv) >= 1:
            target = argv[0].strip('"').strip("'")
            process_directory(target)
        else:
            print("Error: input dir is not specified.")
    else:
        print("Error: use '--' as prefix.")