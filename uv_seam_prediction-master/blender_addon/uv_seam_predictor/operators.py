from types import SimpleNamespace

import os
import shutil

import bpy

from . import export_obj
from . import inference
from . import seam_mapping
from . import validation


def _settings(context):
    return context.scene.uv_seam_predictor_settings


def _ensure_object_mode():
    if bpy.context.object and bpy.context.object.mode == 'OBJECT':
        return
    if not bpy.ops.object.mode_set.poll():
        raise ValueError('Could not switch to Object Mode.')
    bpy.ops.object.mode_set(mode='OBJECT')


def _restore_mode(obj, mode):
    if obj is None or obj.name not in bpy.data.objects:
        return
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if not mode or not bpy.ops.object.mode_set.poll():
        return
    try:
        bpy.ops.object.mode_set(mode=mode)
    except RuntimeError:
        bpy.ops.object.mode_set(mode='OBJECT')


def _resolved_preferences(prefs):
    return SimpleNamespace(
        python_executable=validation.bpy_path_to_os_path(prefs.python_executable),
        predict_script_path=validation.bpy_path_to_os_path(prefs.predict_script_path),
        model_weights_path=validation.bpy_path_to_os_path(prefs.model_weights_path),
        default_timeout_sec=prefs.default_timeout_sec,
        keep_temp_files=prefs.keep_temp_files,
        open_log_on_error=prefs.open_log_on_error,
    )


def _load_error_log(path):
    if not path:
        return
    try:
        text = bpy.data.texts.load(path)
    except RuntimeError:
        return
    text.name = 'UV Seam Predictor Error Log'


class UVSEAM_OT_predict_seams(bpy.types.Operator):
    bl_idname = 'uv_seam_predictor.predict_seams'
    bl_label = 'Predict Seams'
    bl_description = 'Run external seam prediction and apply seam flags'
    bl_options = {'REGISTER', 'UNDO'}

    _job = None
    _timer = None
    _obj = None
    _original_mode = 'OBJECT'
    _prefs = None

    def invoke(self, context, event):
        self._job = None
        self._timer = None
        self._obj = None
        self._original_mode = 'OBJECT'
        self._prefs = None

        settings = _settings(context)
        paths = None
        keep_temp_files = False
        if settings.is_job_running:
            self.report({'WARNING'}, 'UV seam prediction is already running.')
            return {'CANCELLED'}

        try:
            prefs = validation.get_addon_preferences(context)
            keep_temp_files = prefs.keep_temp_files
            validation.validate_configured_paths(prefs)
            obj = validation.require_active_mesh_object(context)
            validation.require_single_user_or_copy_allowed(obj, settings.make_single_user_mesh)
            validation.require_no_enabled_modifiers(obj)
            self._prefs = _resolved_preferences(prefs)

            self._obj = obj
            self._original_mode = obj.mode

            if obj.mode != 'OBJECT':
                _ensure_object_mode()

            validation.require_triangulated_mesh(obj)

            if obj.data.users > 1 and settings.make_single_user_mesh:
                obj.data = obj.data.copy()

            paths = inference.create_temp_work_files()
            export_obj.export_mesh_to_obj(obj, paths['obj_path'])

            self._job = inference.launch_inference(self._prefs, settings, paths)
        except Exception as exc:
            if paths and not keep_temp_files and os.path.isdir(paths['temp_dir']):
                shutil.rmtree(paths['temp_dir'])
            settings.is_job_running = False
            settings.last_run_summary = str(exc)
            _restore_mode(self._obj, self._original_mode)
            self._job = None
            self._obj = None
            self._prefs = None
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        settings.is_job_running = True
        settings.last_run_summary = 'Inference running...'
        try:
            context.window.cursor_set('WAIT')
            self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
            context.window_manager.modal_handler_add(self)
        except Exception as exc:
            if self._timer is not None:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
            if self._job is not None:
                inference.terminate_job(self._job)
                inference.cleanup_job(self._job, keep_temp_files=self._prefs.keep_temp_files if self._prefs else False)
            context.window.cursor_set('DEFAULT')
            _restore_mode(self._obj, self._original_mode)
            settings.is_job_running = False
            settings.last_run_summary = str(exc)
            self._job = None
            self._obj = None
            self._prefs = None
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            return self._finish(context, cancelled=True)

        if event.type != 'TIMER':
            return {'RUNNING_MODAL'}

        if inference.has_timed_out(self._job):
            return self._finish(context, error='Inference timed out.')

        return_code = inference.poll_job(self._job)
        if return_code is None:
            return {'RUNNING_MODAL'}

        inference.close_log_handles(self._job)

        if return_code != 0:
            stderr_tail = inference.read_text_tail(self._job.stderr_path)
            message = f'Inference failed with exit code {return_code}.'
            if stderr_tail:
                message = f'{message} {stderr_tail}'
            return self._finish(context, error=message)

        try:
            predicted_keys = seam_mapping.load_predicted_edge_keys(self._job.json_path)
            result = seam_mapping.apply_seam_keys(
                self._obj.data,
                predicted_keys,
                clear_existing=_settings(context).clear_existing_seams,
            )
            summary = (
                f'Applied {result.applied} seam edges '
                f'({result.missing} missing, {result.duplicates_skipped} duplicates skipped).'
            )
        except Exception as exc:
            return self._finish(context, error=str(exc))

        return self._finish(context, summary=summary)

    def cancel(self, context):
        return self._finish(context, cancelled=True)

    def _finish(self, context, summary=None, error=None, cancelled=False):
        settings = _settings(context)

        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        if self._job is not None:
            if cancelled or self._job.process.poll() is None:
                inference.terminate_job(self._job)
            if error and self._prefs and self._prefs.open_log_on_error:
                inference.close_log_handles(self._job)
                _load_error_log(self._job.stderr_path)
            inference.cleanup_job(self._job, keep_temp_files=self._prefs.keep_temp_files if self._prefs else False)

        context.window.cursor_set('DEFAULT')
        _restore_mode(self._obj, self._original_mode)

        settings.is_job_running = False
        self._job = None
        self._obj = None
        self._prefs = None

        if cancelled:
            settings.last_run_summary = 'Prediction cancelled.'
            self.report({'WARNING'}, settings.last_run_summary)
            return {'CANCELLED'}

        if error:
            settings.last_run_summary = error
            self.report({'ERROR'}, error)
            return {'CANCELLED'}

        settings.last_run_summary = summary or 'Prediction complete.'
        self.report({'INFO'}, settings.last_run_summary)
        return {'FINISHED'}


class UVSEAM_OT_clear_seams(bpy.types.Operator):
    bl_idname = 'uv_seam_predictor.clear_seams'
    bl_label = 'Clear Seams'
    bl_description = 'Clear all seam flags on the active mesh'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = _settings(context)
        obj = None
        original_mode = 'OBJECT'

        try:
            obj = validation.require_active_mesh_object(context)
            original_mode = obj.mode
            if obj.mode != 'OBJECT':
                _ensure_object_mode()

            if obj.data.users > 1:
                if not settings.make_single_user_mesh:
                    raise ValueError('Mesh data is shared. Enable Make Mesh Single User before clearing seams.')
                obj.data = obj.data.copy()

            for edge in obj.data.edges:
                edge.use_seam = False
            obj.data.update()

            summary = f'Cleared seams on {len(obj.data.edges)} edges.'
            settings.last_run_summary = summary
            self.report({'INFO'}, summary)
            return {'FINISHED'}
        except Exception as exc:
            settings.last_run_summary = str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            _restore_mode(obj, original_mode)


class UVSEAM_OT_open_preferences(bpy.types.Operator):
    bl_idname = 'uv_seam_predictor.open_preferences'
    bl_label = 'Open Preferences'
    bl_description = 'Open UV Seam Predictor add-on preferences'

    def execute(self, context):
        bpy.ops.screen.userpref_show('INVOKE_DEFAULT')
        context.preferences.active_section = 'ADDONS'
        if bpy.ops.preferences.addon_show.poll():
            module_name = validation.get_addon_module_name(context)
            bpy.ops.preferences.addon_show(module=module_name)
        return {'FINISHED'}
