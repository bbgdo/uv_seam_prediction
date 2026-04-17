import os


def get_addon_preferences(context):
    package_name = __package__.split('.')[0] if __package__ else 'uv_seam_predictor'
    addon = context.preferences.addons.get(package_name)
    if addon is None:
        addon = context.preferences.addons.get('uv_seam_predictor')
    if addon is None:
        raise ValueError('UV Seam Predictor add-on preferences are not available.')
    return addon.preferences


def require_active_mesh_object(context):
    obj = context.view_layer.objects.active
    if obj is None:
        raise ValueError('Select an active mesh object.')
    if obj.type != 'MESH':
        raise ValueError('Active object must be a mesh.')
    if obj.data is None:
        raise ValueError('Active mesh object has no mesh data.')
    return obj


def require_triangulated_mesh(obj):
    for polygon in obj.data.polygons:
        if len(polygon.vertices) != 3:
            raise ValueError('Mesh must be triangulated before prediction.')


def require_no_enabled_modifiers(obj):
    enabled = [modifier.name for modifier in obj.modifiers if modifier.show_viewport]
    if enabled:
        raise ValueError('Enabled modifiers are not supported for v1. Disable them before prediction.')


def validate_configured_paths(prefs):
    checks = (
        ('Python executable', prefs.python_executable),
        ('Prediction script', prefs.predict_script_path),
        ('Model weights', prefs.model_weights_path),
    )
    for label, path in checks:
        if not path:
            raise ValueError(f'{label} is not configured.')
        if not os.path.exists(bpy_path_to_os_path(path)):
            raise ValueError(f'{label} does not exist: {path}')


def can_make_single_user_mesh(obj):
    return obj.data is not None and obj.data.users > 1


def require_single_user_or_copy_allowed(obj, make_single_user_mesh):
    if obj.data.users > 1 and not make_single_user_mesh:
        raise ValueError('Mesh data is shared. Enable Make Mesh Single User before prediction.')


def bpy_path_to_os_path(path):
    import bpy

    return bpy.path.abspath(path)
