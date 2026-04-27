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
    postprocess_alpha_cost: bpy.props.FloatProperty(
        name='Alpha Cost',
        default=0.50,
        min=0.0,
        max=8.0,
        description='Confidence bias for hop-dominant bridge search cost',
    )
    postprocess_tau_bridge: bpy.props.FloatProperty(
        name='Tau Bridge',
        default=0.20,
        min=0.0,
        max=1.0,
        description='Minimum mean confidence of newly added bridge edges',
    )
    postprocess_conf_floor: bpy.props.FloatProperty(
        name='Conf Floor',
        default=0.10,
        min=0.0,
        max=1.0,
        description='Low-confidence floor used by the bridge acceptance guard',
    )
    postprocess_max_low_conf_fraction: bpy.props.FloatProperty(
        name='Max Low Conf Fraction',
        default=0.50,
        min=0.0,
        max=1.0,
        description='Maximum allowed fraction of new bridge edges below the confidence floor',
    )
    postprocess_force_close_max_edges: bpy.props.IntProperty(
        name='Force Close Max Edges',
        default=5,
        min=0,
        max=16,
        description='Short bridges up to this many new edges bypass the soft bridge acceptance filters',
    )
    postprocess_e0_radius: bpy.props.IntProperty(
        name='E0 Radius',
        default=2,
        min=0,
        max=8,
        description='Local edge-neighborhood radius for pre-bridge thickness collapse',
    )
    postprocess_e0_length_penalty: bpy.props.FloatProperty(
        name='E0 Length Penalty',
        default=0.05,
        min=0.0,
        max=1.0,
        description='Penalty applied per kept edge when choosing the E0 representative path',
    )
    postprocess_r_self: bpy.props.IntProperty(
        name='Self-Bridge Radius',
        default=8,
        min=0,
        max=32,
        description='Maximum new edges used to close a broken loop onto itself',
    )
    postprocess_r_cross: bpy.props.IntProperty(
        name='Cross-Bridge Radius',
        default=10,
        min=0,
        max=64,
        description='Maximum new edges used to attach an open fragment to the main seam graph',
    )
    postprocess_ambiguity_margin: bpy.props.FloatProperty(
        name='Ambiguity Margin',
        default=0.05,
        min=0.0,
        max=2.0,
        description='Minimum mean-confidence gap required to avoid ambiguous bridge choices',
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
    postprocess_max_spur_edges: bpy.props.IntProperty(
        name='Max Spur Edges',
        default=3,
        min=0,
        max=16,
        description='Maximum spur-chain length eligible for post-bridge whisker cleanup',
    )
    postprocess_spur_mean_conf: bpy.props.FloatProperty(
        name='Spur Mean Conf',
        default=0.35,
        min=0.0,
        max=1.0,
        description='Maximum mean confidence allowed for a removable spur chain',
    )
    postprocess_spur_added_fraction_min: bpy.props.FloatProperty(
        name='Spur Added Fraction',
        default=0.50,
        min=0.0,
        max=1.0,
        description='Minimum fraction of spur edges that must come from bridge or collapse stages',
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
