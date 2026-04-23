import bpy


class UVSEAM_Settings(bpy.types.PropertyGroup):
    model_weights_path: bpy.props.StringProperty(
        name='Model Weights Path',
        subtype='FILE_PATH',
        description='Path to the trained model weights file',
    )
    threshold: bpy.props.FloatProperty(
        name='Threshold',
        default=0.75,
        min=0.0,
        max=1.0,
    )
    use_post_processing: bpy.props.BoolProperty(
        name='Apply Post-processing',
        default=True,
        description='Reconnect topological seam fragments and delete floating seam garbage',
    )
    postprocess_seam_threshold: bpy.props.FloatProperty(
        name='Seam Threshold',
        default=0.50,
        min=0.0,
        max=1.0,
        description='Initial seam threshold for the topological post-process',
    )
    postprocess_lambda_off: bpy.props.FloatProperty(
        name='Off-Seam Penalty',
        default=0.75,
        min=0.0,
        max=8.0,
        description='Penalty for stepping through edges below the seam threshold',
    )
    postprocess_r_self: bpy.props.IntProperty(
        name='Self-Bridge Radius',
        default=6,
        min=0,
        max=32,
        description='Maximum new edges used to close a broken loop onto itself',
    )
    postprocess_r_cross: bpy.props.IntProperty(
        name='Cross-Bridge Radius',
        default=8,
        min=0,
        max=64,
        description='Maximum new edges used to attach an open fragment to the main seam graph',
    )
    postprocess_tau_path: bpy.props.FloatProperty(
        name='Path Cost Limit',
        default=1.35,
        min=0.0,
        max=8.0,
        description='Maximum normalized repair-path cost for accepted bridges',
    )
    postprocess_kappa_self: bpy.props.FloatProperty(
        name='Self-Bridge Scale',
        default=1.5,
        min=0.0,
        max=8.0,
        description='Maximum self-bridge path length relative to the broken arc length',
    )
    postprocess_attach_margin: bpy.props.FloatProperty(
        name='Attach Margin',
        default=0.10,
        min=0.0,
        max=2.0,
        description='Required cost gap between the best and second-best main-graph attachments',
    )
    postprocess_garbage_max_edges: bpy.props.IntProperty(
        name='Garbage Max Edges',
        default=4,
        min=0,
        max=32,
        description='Open fragments up to this size are treated as disposable garbage if repair fails',
    )
    postprocess_r_snap: bpy.props.IntProperty(
        name='Snap Radius',
        default=3,
        min=0,
        max=16,
        description='Maximum hop distance for classifying a fragment as near-main during band collapse',
    )
    postprocess_snap_max_edges: bpy.props.IntProperty(
        name='Snap Max Edges',
        default=12,
        min=0,
        max=64,
        description='Maximum fragment size eligible for band-collapse snapping',
    )
    postprocess_r_band: bpy.props.IntProperty(
        name='Band Radius',
        default=2,
        min=0,
        max=16,
        description='Dilation radius used to build the local band around the main seam and satellite fragment',
    )
    postprocess_eta_main: bpy.props.FloatProperty(
        name='Stay-Near-Main Weight',
        default=0.35,
        min=0.0,
        max=8.0,
        description='Penalty for drifting away from the current main seam during band collapse',
    )
    clear_existing_seams: bpy.props.BoolProperty(
        name='Clear Existing Seams',
        default=True,
    )
    make_single_user_mesh: bpy.props.BoolProperty(
        name='Make Mesh Single User',
        default=True,
    )
    last_run_summary: bpy.props.StringProperty(
        name='Last Run Summary',
        default='',
    )
    is_job_running: bpy.props.BoolProperty(
        name='Job Running',
        default=False,
    )
