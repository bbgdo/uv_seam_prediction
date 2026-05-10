bl_info = {
    'name': 'ML Seams',
    'author': 'ML Seams',
    'version': (0, 1, 0),
    'blender': (4, 0, 0),
    'location': 'View3D > Sidebar > Auto Seams',
    'description': 'Automatically mark UV seam edges with an external inference script.',
    'category': 'UV',
}

import bpy
import importlib
import sys

should_reload = f'{__name__}.prefs' in sys.modules

from . import export_obj
from . import inference
from . import operators
from . import prefs
from . import properties
from . import seam_mapping
from . import ui
from . import validation

reload_modules = (
    prefs,
    properties,
    validation,
    export_obj,
    inference,
    seam_mapping,
    operators,
    ui,
)
if should_reload:
    for module in reload_modules:
        importlib.reload(module)


classes = (
    prefs.UVSEAM_Preferences,
    properties.UVSEAM_Settings,
    operators.UVSEAM_OT_open_preferences,
    operators.UVSEAM_OT_clear_seams,
    operators.UVSEAM_OT_fill_current_seam_gaps,
    operators.UVSEAM_OT_clean_small_dangling_seams,
    operators.UVSEAM_OT_mirror_current_seams_left_to_right,
    operators.UVSEAM_OT_mirror_current_seams_right_to_left,
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
