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
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')


def _resolved_preferences(prefs):
    return SimpleNamespace(
        python_executable=validation.bpy_path_to_os_path(prefs.python_executable),
        predict_script_path=validation.bpy_path_to_os_path(prefs.predict_script_path),
        default_timeout_sec=prefs.default_timeout_sec,
        keep_temp_files=prefs.keep_temp_files,
        open_log_on_error=prefs.open_log_on_error,
    )


def _resolved_run_settings(settings):
    return SimpleNamespace(
        model_weights_path=validation.bpy_path_to_os_path(settings.model_weights_path),
        threshold=settings.threshold,
        use_post_processing=settings.use_post_processing,
        postprocess_tau_low=settings.postprocess_tau_low,
        postprocess_tau_high=settings.postprocess_tau_high,
        postprocess_d_max=settings.postprocess_d_max,
        postprocess_r_bridge=settings.postprocess_r_bridge,
        postprocess_l_min=settings.postprocess_l_min,
        postprocess_epsilon=settings.postprocess_epsilon,
        postprocess_anchor_boundary=settings.postprocess_anchor_boundary,
        postprocess_fill_small_gaps=settings.postprocess_fill_small_gaps,
        postprocess_fill_gap_max_hops=settings.postprocess_fill_gap_max_hops,
        postprocess_write_debug_sidecars=settings.postprocess_write_debug_sidecars,
        clear_existing_seams=settings.clear_existing_seams,
    )


def _mesh_topology_counts(mesh):
    return (len(mesh.vertices), len(mesh.edges))


def _load_error_log(path):
    if not path:
        return
    try:
        text = bpy.data.texts.load(path)
    except RuntimeError:
        return
    text.name = 'Auto Seams Error Log'


def _cleanup_suffix(errors):
    if not errors:
        return ''
    return ' Cleanup issues: ' + '; '.join(errors)


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
    _run_settings = None
    _topology_counts = None

    def invoke(self, context, event):
        self._job = None
        self._timer = None
        self._obj = None
        self._original_mode = 'OBJECT'
        self._prefs = None
        self._run_settings = None
        self._topology_counts = None

        settings = _settings(context)
        paths = None
        keep_temp_files = False
        if settings.is_job_running:
            self.report({'WARNING'}, 'Auto Seams prediction is already running.')
            return {'CANCELLED'}

        try:
            prefs = validation.get_addon_preferences(context)
            keep_temp_files = prefs.keep_temp_files
            validation.validate_configured_paths(prefs, settings)
            obj = validation.require_active_mesh_object(context)
            validation.require_single_user_or_copy_allowed(obj, settings.make_single_user_mesh)
            validation.require_no_enabled_modifiers(obj)
            self._prefs = _resolved_preferences(prefs)
            self._run_settings = _resolved_run_settings(settings)

            self._obj = obj
            self._original_mode = obj.mode

            if obj.mode != 'OBJECT':
                _ensure_object_mode()

            if obj.data.users > 1 and settings.make_single_user_mesh:
                obj.data = obj.data.copy()

            self._topology_counts = _mesh_topology_counts(obj.data)

            paths = inference.create_temp_work_files()
            export_obj.export_object_to_obj_with_hidden_triangulation(obj, paths['obj_path'])

            self._job = inference.launch_inference(self._prefs, self._run_settings, paths)
        except Exception as exc:
            cleanup_errors = []
            if paths and not keep_temp_files and os.path.isdir(paths['temp_dir']):
                try:
                    shutil.rmtree(paths['temp_dir'])
                except Exception as cleanup_exc:
                    cleanup_errors.append(f'temp files not removed: {cleanup_exc}')
            try:
                _restore_mode(self._obj, self._original_mode)
            except Exception as restore_exc:
                cleanup_errors.append(f'mode not restored: {restore_exc}')
            message = f'{exc}{_cleanup_suffix(cleanup_errors)}'
            settings.is_job_running = False
            settings.last_run_summary = message
            self._job = None
            self._obj = None
            self._prefs = None
            self._run_settings = None
            self._topology_counts = None
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        settings.is_job_running = True
        settings.last_run_summary = 'Inference running...'
        try:
            context.window.cursor_set('WAIT')
            self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
            context.window_manager.modal_handler_add(self)
        except Exception as exc:
            return self._finish(context, error=str(exc))
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
            if self._obj is None or self._obj.name not in bpy.data.objects:
                raise ValueError('Active mesh was removed while prediction was running. No seams were applied.')
            validation.require_unchanged_topology(self._obj, self._topology_counts)
            predicted_keys = seam_mapping.load_predicted_edge_keys(self._job.json_path)
            accepted_bridge_entries = seam_mapping.load_accepted_bridge_debug_entries(self._job.json_path)
            result = seam_mapping.apply_seam_keys(
                self._obj.data,
                predicted_keys,
                clear_existing=self._run_settings.clear_existing_seams,
                accepted_bridge_entries=accepted_bridge_entries,
                enable_local_repair=self._run_settings.use_post_processing,
                fill_small_gaps=self._run_settings.postprocess_fill_small_gaps,
                fill_gap_max_hops=self._run_settings.postprocess_fill_gap_max_hops,
                collect_debug_diagnostics=self._run_settings.postprocess_write_debug_sidecars,
            )
            summary = seam_mapping.format_apply_summary(result)
            if self._run_settings.postprocess_write_debug_sidecars:
                debug_path = seam_mapping.write_bridge_apply_debug(self._job.json_path, result)
                human_gap_path = seam_mapping.write_human_gap_classification(self._job.json_path, result)
                residual_debug_path = seam_mapping.write_residual_gap_phase2e_debug(
                    self._job.json_path,
                    result,
                )
                residual_summary = seam_mapping.format_residual_gap_phase2e_summary(
                    result.residual_gap_phase2e_debug or {},
                    residual_debug_path,
                )
                ranking_debug = seam_mapping.build_endpoint_bridge_ranking_debug(result)
                ranking_debug_path = seam_mapping.write_endpoint_bridge_ranking_debug(
                    self._job.json_path,
                    result,
                )
                ranking_summary = seam_mapping.format_endpoint_bridge_ranking_debug_summary(
                    ranking_debug,
                    ranking_debug_path,
                )
                rank_review = seam_mapping.build_rank_9_to_16_review(result)
                rank_review_path = seam_mapping.write_rank_9_to_16_review(
                    self._job.json_path,
                    result,
                )
                rank_review_summary = seam_mapping.format_rank_9_to_16_review_summary(
                    rank_review,
                    rank_review_path,
                )
                general_candidates_path = seam_mapping.write_general_residual_candidates_phase2h(
                    self._job.json_path,
                    result,
                )
                general_candidates_summary = seam_mapping.format_general_residual_candidates_phase2h_summary(
                    result.general_residual_candidates_phase2h or {},
                    general_candidates_path,
                )
                unified_simulation_path = seam_mapping.write_unified_local_continuity_simulation_phase2h_r(
                    self._job.json_path,
                    result,
                )
                unified_simulation_summary = seam_mapping.format_unified_local_continuity_simulation_phase2h_r_summary(
                    result.unified_local_continuity_simulation_phase2h_r or {},
                    unified_simulation_path,
                )
                visual_review_path = seam_mapping.write_phase2h_r3_visual_review(
                    self._job.json_path,
                    result,
                )
                visual_review_summary = seam_mapping.format_phase2h_r3_visual_review_summary(
                    result.phase2h_r3_visual_review or {},
                    visual_review_path,
                )
                small_gap_simulation_path = seam_mapping.write_phase2j_r_small_gap_rule_simulation(
                    self._job.json_path,
                    result,
                )
                small_gap_simulation_summary = seam_mapping.format_phase2j_r_small_gap_rule_simulation_summary(
                    result.phase2j_r_small_gap_rule_simulation or {},
                    small_gap_simulation_path,
                )
                tangent_audit_path = seam_mapping.write_phase2k_r_tangent_audit_rescue(
                    self._job.json_path,
                    result,
                )
                tangent_audit_summary = seam_mapping.format_phase2k_r_tangent_audit_rescue_summary(
                    result.phase2k_r_tangent_audit_rescue or {},
                    tangent_audit_path,
                )
                summary = (
                    f'{summary} Bridge apply debug: {debug_path}. '
                    f'Human gap classification: {human_gap_path}. '
                    f'{residual_summary}. '
                    f'{ranking_summary}. '
                    f'{rank_review_summary}. '
                    f'{general_candidates_summary}. '
                    f'{unified_simulation_summary}. '
                    f'{visual_review_summary}. '
                    f'{small_gap_simulation_summary}. '
                    f'{tangent_audit_summary}'
                )
        except Exception as exc:
            return self._finish(context, error=str(exc))

        return self._finish(context, summary=summary)

    def cancel(self, context):
        return self._finish(context, cancelled=True)

    def _finish(self, context, summary=None, error=None, cancelled=False):
        settings = _settings(context)
        cleanup_errors = []

        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception as exc:
                cleanup_errors.append(f'timer not removed: {exc}')
            finally:
                self._timer = None

        if self._job is not None:
            try:
                if cancelled or self._job.process.poll() is None:
                    inference.terminate_job(self._job)
            except Exception as exc:
                cleanup_errors.append(f'process not terminated cleanly: {exc}')
            try:
                if error and self._prefs and self._prefs.open_log_on_error:
                    inference.close_log_handles(self._job)
                    _load_error_log(self._job.stderr_path)
            except Exception as exc:
                cleanup_errors.append(f'error log not opened: {exc}')
            try:
                keep_temp_files = self._prefs.keep_temp_files if self._prefs else False
                inference.cleanup_job(self._job, keep_temp_files=keep_temp_files)
            except Exception as exc:
                cleanup_errors.append(f'temp files not removed: {exc}')

        try:
            context.window.cursor_set('DEFAULT')
        except Exception as exc:
            cleanup_errors.append(f'cursor not restored: {exc}')
        try:
            _restore_mode(self._obj, self._original_mode)
        except Exception as exc:
            cleanup_errors.append(f'mode not restored: {exc}')

        settings.is_job_running = False
        self._job = None
        self._obj = None
        self._prefs = None
        self._run_settings = None
        self._topology_counts = None

        if cancelled:
            settings.last_run_summary = f'Prediction cancelled.{_cleanup_suffix(cleanup_errors)}'
            self.report({'WARNING'}, settings.last_run_summary)
            return {'CANCELLED'}

        if error:
            settings.last_run_summary = f'{error}{_cleanup_suffix(cleanup_errors)}'
            self.report({'ERROR'}, settings.last_run_summary)
            return {'CANCELLED'}

        final_summary = summary or 'Prediction complete.'
        settings.last_run_summary = f'{final_summary}{_cleanup_suffix(cleanup_errors)}'
        self.report({'WARNING' if cleanup_errors else 'INFO'}, settings.last_run_summary)
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


class UVSEAM_OT_fill_current_seam_gaps(bpy.types.Operator):
    bl_idname = 'uv_seam_predictor.fill_current_seam_gaps'
    bl_label = 'Fill Gaps on Current Seams'
    bl_description = 'Fill small gaps in the currently marked seam edges using editable mesh topology'
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

            result = seam_mapping.apply_editable_shortest_path_gap_fill(
                obj.data,
                enabled=True,
                max_gap_hops=settings.postprocess_fill_gap_max_hops,
                allow_same_component=False,
            )
            obj.data.update()

            paths = int(result.get('accepted_paths_count', 0))
            edges = int(result.get('accepted_edges_count', 0))
            max_hops = int(result.get('max_gap_hops', settings.postprocess_fill_gap_max_hops))
            if paths:
                summary = f'Filled {paths} seam gap paths / {edges} edges with max hops {max_hops}.'
            else:
                summary = f'No seam gaps filled with max hops {max_hops}.'
            settings.last_run_summary = summary
            self.report({'INFO'}, summary)
            return {'FINISHED'}
        except Exception as exc:
            message = str(exc)
            settings.last_run_summary = message
            self.report({'WARNING'}, message)
            return {'CANCELLED'}
        finally:
            _restore_mode(obj, original_mode)


class UVSEAM_OT_clean_small_dangling_seams(bpy.types.Operator):
    bl_idname = 'uv_seam_predictor.clean_small_dangling_seams'
    bl_label = 'Clean Small Dangling Seams'
    bl_description = 'Remove short dangling seam branches from the currently marked seams'
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

            result = seam_mapping.apply_editable_dangling_seam_cleanup(
                obj.data,
                enabled=True,
                max_dangling_edges=settings.manual_cleanup_max_dangling_edges,
                protect_boundary_vertices=settings.manual_cleanup_protect_boundary_vertices,
                allow_remove_entire_component=False,
            )
            obj.data.update()

            branches = int(result.get('removed_branches_count', 0))
            edges = int(result.get('removed_edges_count', 0))
            max_edges = int(result.get(
                'max_dangling_edges',
                settings.manual_cleanup_max_dangling_edges,
            ))
            if branches:
                summary = (
                    f'Removed {branches} dangling seam branches / {edges} edges '
                    f'with max length {max_edges}.'
                )
            else:
                summary = f'No dangling seam branches removed with max length {max_edges}.'
            settings.last_run_summary = summary
            self.report({'INFO'}, summary)
            return {'FINISHED'}
        except Exception as exc:
            message = str(exc)
            settings.last_run_summary = message
            self.report({'WARNING'}, message)
            return {'CANCELLED'}
        finally:
            _restore_mode(obj, original_mode)


def _execute_manual_seam_mirror(operator, context, direction, label):
    settings = _settings(context)
    obj = None
    original_mode = 'OBJECT'

    try:
        obj = validation.require_active_mesh_object(context)
        original_mode = obj.mode
        if obj.mode != 'OBJECT':
            _ensure_object_mode()

        result = seam_mapping.apply_editable_seam_mirror(
            obj.data,
            enabled=True,
            direction=direction,
            axis=settings.manual_mirror_axis,
            tolerance=settings.manual_mirror_tolerance,
            skip_center_edges=True,
        )
        obj.data.update()

        axis = str(result.get('axis', settings.manual_mirror_axis)).upper()
        added = int(result.get('mirrored_edges_added', 0))
        already = int(result.get('mirrored_edges_already_present', 0))
        source = int(result.get('source_seam_edges', 0))
        unmatched = int(result.get('unmatched_vertices', 0))
        missing = int(result.get('missing_mirrored_edges', 0))
        skipped = int(result.get('skipped_center_edges', 0))
        summary = (
            f'Mirror {axis} {label}: added {added}, already {already}, source {source}, '
            f'unmatched vertices {unmatched}, missing edges {missing}, skipped center {skipped}.'
        )
        settings.last_run_summary = summary
        operator.report({'INFO'}, summary)
        return {'FINISHED'}
    except Exception as exc:
        message = str(exc)
        settings.last_run_summary = message
        operator.report({'WARNING'}, message)
        return {'CANCELLED'}
    finally:
        _restore_mode(obj, original_mode)


class UVSEAM_OT_mirror_current_seams_left_to_right(bpy.types.Operator):
    bl_idname = 'uv_seam_predictor.mirror_current_seams_l_to_r'
    bl_label = 'Mirror Seams −→+'
    bl_description = (
        'Mirror current seam flags from the negative side of the selected local axis '
        'to the positive side'
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return _execute_manual_seam_mirror(self, context, 'NEGATIVE_TO_POSITIVE', '−→+')


class UVSEAM_OT_mirror_current_seams_right_to_left(bpy.types.Operator):
    bl_idname = 'uv_seam_predictor.mirror_current_seams_r_to_l'
    bl_label = 'Mirror Seams +→−'
    bl_description = (
        'Mirror current seam flags from the positive side of the selected local axis '
        'to the negative side'
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return _execute_manual_seam_mirror(self, context, 'POSITIVE_TO_NEGATIVE', '+→−')


class UVSEAM_OT_open_preferences(bpy.types.Operator):
    bl_idname = 'uv_seam_predictor.open_preferences'
    bl_label = 'Open Preferences'
    bl_description = 'Open Auto Seams add-on preferences'

    def execute(self, context):
        bpy.ops.screen.userpref_show('INVOKE_DEFAULT')
        context.preferences.active_section = 'ADDONS'
        if bpy.ops.preferences.addon_show.poll():
            module_name = validation.get_addon_module_name(context)
            bpy.ops.preferences.addon_show(module=module_name)
        return {'FINISHED'}
