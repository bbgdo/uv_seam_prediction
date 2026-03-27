import bpy
import os
import sys
import shutil

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

    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images, bpy.data.armatures,
                       bpy.data.libraries, bpy.data.lights, bpy.data.cameras]:
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


def import_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if not os.path.exists(filepath):
        print(f"CRITICAL: file not found: {filepath}")
        return False

    try:
        if ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=filepath)
        elif ext == '.dae':
            bpy.ops.wm.collada_import(filepath=filepath)
        elif ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=filepath)
        elif ext == '.blend':
            with bpy.data.libraries.load(filepath) as (data_from, data_to):
                data_to.objects = data_from.objects

            found_objects = False
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)
                    obj.make_local()
                    found_objects = True

            if not found_objects:
                print(f"WARN: no objects in {filepath}")
                return False
        else:
            return False
        return True
    except Exception as e:
        print(f"FAIL: import {os.path.basename(filepath)}: {e}")
        return False


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
        # Blender < 4.0
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
    output_dir = os.path.join(parent_dir, f"{base_name}_OBJ")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    already_obj_dir = os.path.join(output_dir, "already_obj")
    if not os.path.exists(already_obj_dir):
        os.makedirs(already_obj_dir)

    valid_extensions = {'.fbx', '.dae', '.glb', '.gltf', '.blend', '.obj'}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

    print(f"\n=== Starting ===")
    print(f"Input dir: {input_dir}")
    print(f"Export dir: {output_dir}")
    print(f"Found files: {len(files)}")
    print("======================\n")

    success_count = 0
    fail_count = 0
    copied_count = 0

    for f in files:
        full_path = os.path.join(input_dir, f)
        filename_only = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[1].lower()

        if ext == '.obj':
            sys.stdout.write(f"Copying OBJ: {f} ... ")
            try:
                destination = os.path.join(already_obj_dir, f)
                shutil.copy2(full_path, destination)
                print("COPIED")
                copied_count += 1
            except Exception as e:
                print(f"ERROR COPYING: {e}")
                fail_count += 1
            continue

        sys.stdout.write(f"Converting: {f} ... ")
        sys.stdout.flush()

        reset_scene()

        if import_file(full_path):
            bpy.ops.object.select_all(action='SELECT')

            for obj in bpy.context.selected_objects:
                if obj.type != 'MESH':
                    obj.select_set(False)

            if bpy.context.selected_objects:
                target_file = f"{filename_only}.obj"
                out_path = get_unique_filepath(output_dir, target_file)

                try:
                    export_obj_modern(out_path)
                    print("OK")
                    success_count += 1
                except Exception as e:
                    print(f"FAIL (Export): {e}")
                    fail_count += 1
            else:
                print("FAIL (No meshes found)")
                fail_count += 1
        else:
            print("FAIL (Import)")
            fail_count += 1

    print(f"\n=== Finished ===")
    print(f"Converted: {success_count}")
    print(f"Copied (already OBJ): {copied_count}")
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
        print("Error: use '--' as prefix for input dir argument.")