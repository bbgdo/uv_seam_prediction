import bpy
import os
import sys

EXPORT_CONFIG = {
    "use_selection": True,
    "use_mesh_modifiers": True,
    "use_normals": True,
    "use_uvs": True,
    "use_triangles": False,
    "use_materials": False,
    "axis_forward": '-Z',
    "axis_up": 'Y',
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


def robust_cleanup_and_triangulate():
    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not mesh_objects:
        return False

    # Make the first object active (required for Edit Mode)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    # 1. Enter Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # --- CLEANUP STAGE ---
    # a) Merge by Distance (replaces Remove Doubles)
    # Merges vertices that are very close to each other
    bpy.ops.mesh.remove_doubles(threshold=0.0001)

    # b) Delete Loose (removes loose vertices and edges)
    bpy.ops.mesh.delete_loose()

    # c) Dissolve Degenerate (removes geometry with zero area/length)
    bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)

    # --- NORMALS STAGE ---
    # Recalculate normals to face outward (important for correct shading)
    bpy.ops.mesh.normals_make_consistent(inside=False)

    # --- TRIANGULATION STAGE ---
    # Converts all polygons to triangles (most stable method)
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')


def export_obj_modern(out_path):
    try:
        # Blender 4.0+ API
        bpy.ops.wm.obj_export(
            filepath=out_path,
            export_selected_objects=True,
            apply_modifiers=EXPORT_CONFIG["use_mesh_modifiers"],
            export_uv=EXPORT_CONFIG["use_uvs"],
            export_normals=EXPORT_CONFIG["use_normals"],
            export_triangulated_mesh=EXPORT_CONFIG["use_triangles"],
            export_materials=False,
            forward_axis='NEGATIVE_Z',
            up_axis='Y'
        )
    except AttributeError:
        # Legacy API
        bpy.ops.export_scene.obj(
            filepath=out_path,
            **EXPORT_CONFIG
        )


def process_directory(input_path_arg):
    input_dir = os.path.abspath(input_path_arg)
    if not os.path.exists(input_dir):
        print(f"Error: Dir '{input_dir}' is not found.")
        return

    # Build output folder path (input_machine_cleaned)
    parent_dir = os.path.dirname(input_dir.rstrip(os.sep))
    base_name = os.path.basename(input_dir.rstrip(os.sep))
    output_dir = os.path.join(parent_dir, f"{base_name}_machine_cleaned")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.obj')]

    print(f"\n=== Geometry Cleanup Pipeline (Native) ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files: {len(files)}")
    print("=========================================\n")

    success_count = 0
    fail_count = 0

    for f in files:
        full_path = os.path.join(input_dir, f)

        # Print progress without newline
        sys.stdout.write(f"Processing: {f} ... ")
        sys.stdout.flush()

        reset_scene()

        # Import
        try:
            if hasattr(bpy.ops.wm, "obj_import"):
                bpy.ops.wm.obj_import(filepath=full_path)
            else:
                bpy.ops.import_scene.obj(filepath=full_path)
        except Exception as e:
            print(f"FAIL (Import Error)")
            fail_count += 1
            continue

        # Process
        try:
            if robust_cleanup_and_triangulate():
                target_file = f
                out_path = get_unique_filepath(output_dir, target_file)
                export_obj_modern(out_path)
                print("OK")
                success_count += 1
            else:
                print("FAIL (No geometry found)")
                fail_count += 1
        except Exception as e:
            print(f"FAIL (Processing Error: {e})")
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