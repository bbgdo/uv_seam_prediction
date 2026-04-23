import os

import bpy

from . import validation


def _path_state(path):
    if not path:
        return 'Not set', 'ERROR'
    if os.path.exists(validation.bpy_path_to_os_path(path)):
        return 'OK', 'CHECKMARK'
    return 'Missing', 'ERROR'


class UVSEAM_PT_panel(bpy.types.Panel):
    bl_idname = 'UVSEAM_PT_panel'
    bl_label = 'Auto Seams'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Auto Seams'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.uv_seam_predictor_settings

        try:
            prefs = validation.get_addon_preferences(context)
        except ValueError:
            prefs = None

        setup = layout.box()
        setup.label(text='Setup')
        if prefs:
            for label, path in (
                ('Python', prefs.python_executable),
                ('Script', prefs.predict_script_path),
                ('Model', settings.model_weights_path),
            ):
                status, icon = _path_state(path)
                row = setup.row()
                row.label(text=f'{label}: {status}', icon=icon)
        else:
            setup.label(text='Preferences unavailable', icon='ERROR')
        setup.operator('uv_seam_predictor.open_preferences', icon='PREFERENCES')

        inference_box = layout.box()
        inference_box.label(text='Inference')
        inference_box.prop(settings, 'model_weights_path')
        inference_box.prop(settings, 'threshold')
        inference_box.prop(settings, 'use_post_processing')
        inference_box.prop(settings, 'clear_existing_seams')
        inference_box.prop(settings, 'make_single_user_mesh')

        post_box = layout.box()
        post_box.label(text='Post-process Settings')
        post_box.enabled = settings.use_post_processing
        post_box.prop(settings, 'postprocess_seam_threshold')
        post_box.prop(settings, 'postprocess_lambda_off')
        post_box.prop(settings, 'postprocess_r_self')
        post_box.prop(settings, 'postprocess_r_cross')
        post_box.prop(settings, 'postprocess_tau_path')
        post_box.prop(settings, 'postprocess_kappa_self')
        post_box.prop(settings, 'postprocess_attach_margin')
        post_box.prop(settings, 'postprocess_garbage_max_edges')
        post_box.prop(settings, 'postprocess_r_snap')
        post_box.prop(settings, 'postprocess_snap_max_edges')
        post_box.prop(settings, 'postprocess_r_band')
        post_box.prop(settings, 'postprocess_eta_main')

        actions = layout.box()
        actions.label(text='Actions')
        row = actions.row()
        row.enabled = not settings.is_job_running
        row.operator('uv_seam_predictor.predict_seams', icon='PLAY')
        row = actions.row()
        row.enabled = not settings.is_job_running
        row.operator('uv_seam_predictor.clear_seams', icon='X')

        status = layout.box()
        status.label(text='Status')
        status.label(text='Running' if settings.is_job_running else 'Idle')
        if settings.last_run_summary:
            status.label(text=settings.last_run_summary)
