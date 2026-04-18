import bpy


class UVSEAM_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or 'uv_seam_predictor'

    python_executable: bpy.props.StringProperty(
        name='Python Executable',
        subtype='FILE_PATH',
        description='Python executable used to run the seam prediction script',
    )
    predict_script_path: bpy.props.StringProperty(
        name='Prediction Script',
        subtype='FILE_PATH',
        description='Path to tools/predict_seams.py',
    )
    default_timeout_sec: bpy.props.IntProperty(
        name='Timeout',
        default=300,
        min=1,
        description='Maximum inference runtime in seconds',
    )
    keep_temp_files: bpy.props.BoolProperty(
        name='Keep Temporary Files',
        default=False,
        description='Keep exported OBJ, JSON, and process logs after inference',
    )
    open_log_on_error: bpy.props.BoolProperty(
        name='Open Log On Error',
        default=False,
        description='Open stderr log in Blender text editor when inference fails',
    )

    def draw(self, context):
        layout = self.layout

        paths = layout.box()
        paths.label(text='External Inference')
        paths.prop(self, 'python_executable')
        paths.prop(self, 'predict_script_path')

        runtime = layout.box()
        runtime.label(text='Runtime')
        runtime.prop(self, 'default_timeout_sec')
        runtime.prop(self, 'keep_temp_files')
        runtime.prop(self, 'open_log_on_error')
