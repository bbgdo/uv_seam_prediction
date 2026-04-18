import os


def addon_module_candidates():
    package_name = __package__ or 'uv_seam_predictor'
    candidates = [package_name, 'uv_seam_predictor']
    if '.' in package_name:
        candidates.append(package_name.split('.')[0])
    return tuple(dict.fromkeys(candidates))


def get_addon_module_name(context):
    for module_name in addon_module_candidates():
        if module_name in context.preferences.addons:
            return module_name
    raise ValueError('Auto Seams add-on preferences are not available.')


def get_addon_preferences(context):
    return context.preferences.addons[get_addon_module_name(context)].preferences


def require_active_mesh_object(context):
    obj = context.view_layer.objects.active
    if obj is None:
        raise ValueError('Select an active mesh object.')
    if obj.type != 'MESH':
        raise ValueError('Active object must be a mesh.')
    if obj.data is None:
        raise ValueError('Active mesh object has no mesh data.')
    return obj


def require_no_enabled_modifiers(obj):
    enabled = [modifier.name for modifier in obj.modifiers if modifier.show_viewport]
    if enabled:
        raise ValueError('Enabled modifiers are not supported for v1. Disable them before prediction.')


def validate_configured_paths(prefs, settings):
    checks = (
        ('Python executable', prefs.python_executable),
        ('Prediction script', prefs.predict_script_path),
        ('Model weights', settings.model_weights_path),
    )
    for label, path in checks:
        if not path:
            raise ValueError(f'{label} is not configured.')
        if not os.path.exists(bpy_path_to_os_path(path)):
            raise ValueError(f'{label} does not exist: {path}')


def require_single_user_or_copy_allowed(obj, make_single_user_mesh):
    if obj.data.users > 1 and not make_single_user_mesh:
        raise ValueError('Mesh data is shared. Enable Make Mesh Single User before prediction.')


def require_unchanged_topology(obj, expected_counts):
    current_counts = (len(obj.data.vertices), len(obj.data.edges))
    if current_counts != expected_counts:
        raise ValueError('Mesh topology changed while prediction was running. No seams were applied.')


def bpy_path_to_os_path(path):
    import bpy

    return bpy.path.abspath(path)
