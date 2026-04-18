bl_info = {
    'name': 'Auto Seams',
    'author': 'Auto Seams',
    'version': (0, 1, 0),
    'blender': (4, 0, 0),
    'location': 'View3D > Sidebar > Auto Seams',
    'description': 'Automatically mark UV seam edges with an external inference script.',
    'category': 'UV',
}

import bpy

if 'prefs' in locals():
    import importlib

    importlib.reload(prefs)
    importlib.reload(properties)
    importlib.reload(validation)
    importlib.reload(export_obj)
    importlib.reload(inference)
    importlib.reload(seam_mapping)
    importlib.reload(operators)
    importlib.reload(ui)
else:
    from . import export_obj
    from . import inference
    from . import operators
    from . import prefs
    from . import properties
    from . import seam_mapping
    from . import ui
    from . import validation


classes = (
    prefs.UVSEAM_Preferences,
    properties.UVSEAM_Settings,
    operators.UVSEAM_OT_open_preferences,
    operators.UVSEAM_OT_clear_seams,
    operators.UVSEAM_OT_predict_seams,
    ui.UVSEAM_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.uv_seam_predictor_settings = bpy.props.PointerProperty(
        type=properties.UVSEAM_Settings,
    )


def unregister():
    if hasattr(bpy.types.Scene, 'uv_seam_predictor_settings'):
        del bpy.types.Scene.uv_seam_predictor_settings

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
