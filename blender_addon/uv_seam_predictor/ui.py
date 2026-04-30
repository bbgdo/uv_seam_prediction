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
        inference_box.label(text='Threshold controls raw seam classification.', icon='INFO')
        inference_box.label(text='Post-processing may skeletonize, bridge, prune, and fill editable gaps.')
        inference_box.prop(settings, 'clear_existing_seams')
        inference_box.prop(settings, 'make_single_user_mesh')

        post_box = layout.box()
        post_box.label(text='Post-processing')
        post_box.enabled = settings.use_post_processing
        post_box.prop(settings, 'postprocess_tau_low')
        post_box.prop(settings, 'postprocess_d_max')
        post_box.prop(settings, 'postprocess_r_bridge')
        post_box.prop(settings, 'postprocess_l_min')
        post_box.prop(settings, 'postprocess_anchor_boundary')
        post_box.prop(settings, 'postprocess_fill_small_gaps')
        gap_row = post_box.row()
        gap_row.enabled = settings.postprocess_fill_small_gaps
        gap_row.prop(settings, 'postprocess_fill_gap_max_hops')

        manual_box = layout.box()
        manual_box.label(text='Manual Seam Cleanup')
        manual_box.operator('uv_seam_predictor.fill_current_seam_gaps', icon='MOD_VERTEX_WEIGHT')
        manual_box.prop(settings, 'postprocess_fill_gap_max_hops')
        manual_box.operator('uv_seam_predictor.clean_small_dangling_seams', icon='BRUSH_DATA')
        manual_box.prop(settings, 'manual_cleanup_max_dangling_edges')
        manual_box.prop(settings, 'manual_cleanup_protect_boundary_vertices')

        legacy_box = layout.box()
        legacy_box.label(text='Legacy / Debug')
        legacy_box.enabled = settings.use_post_processing
        legacy_box.prop(settings, 'postprocess_tau_high')
        legacy_box.prop(settings, 'postprocess_epsilon')
        legacy_box.prop(settings, 'postprocess_write_debug_sidecars')

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
