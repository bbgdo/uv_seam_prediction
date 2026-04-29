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
    postprocess_tau_low: bpy.props.FloatProperty(
        name='Tau Low (Stage A candidates)',
        default=0.30,
        min=0.0,
        max=1.0,
        description='Lower probability threshold defining the candidate vertex set '
                    'for skeletonization (Stage A).',
    )
    postprocess_tau_high: bpy.props.FloatProperty(
        name='Tau High (Stage B terminals)',
        default=0.70,
        min=0.0,
        max=1.0,
        description='Upper probability threshold defining confidence-based Steiner '
                    'terminals for bridging (Stage B).',
    )
    postprocess_d_max: bpy.props.IntProperty(
        name='D Max (Stage A thickness)',
        default=3,
        min=1,
        max=16,
        description='Maximum allowed distance between any thinned vertex and the '
                    'preserved skeleton, in mesh edges (Stage A).',
    )
    postprocess_r_bridge: bpy.props.IntProperty(
        name='R Bridge (Stage B radius)',
        default=6,
        min=0,
        max=32,
        description='Bounded search radius for Steiner bridging within each skeleton '
                    'component, in mesh edges (Stage B). 0 disables bridging.',
    )
    postprocess_l_min: bpy.props.IntProperty(
        name='L Min (Stage C spur length)',
        default=4,
        min=1,
        max=32,
        description='Minimum branch length retained during spur pruning, in mesh '
                    'edges (Stage C). Shorter dangling branches are removed.',
    )
    postprocess_epsilon: bpy.props.FloatProperty(
        name='Epsilon (numerical floor)',
        default=1e-3,
        min=1e-6,
        max=0.1,
        precision=6,
        description='Numerical floor used in -log(p) edge weighting to avoid '
                    'underflow on near-zero probabilities (Stage B).',
    )
    postprocess_anchor_boundary: bpy.props.BoolProperty(
        name='Anchor Mesh Boundary',
        default=True,
        description='If enabled, mesh boundary vertices are treated as structural '
                    'anchors throughout all three stages.',
    )
    postprocess_fill_small_gaps: bpy.props.BoolProperty(
        name='Fill Small Gaps',
        default=True,
        description='Fill small seam gaps on the editable Blender mesh after prediction.',
    )
    postprocess_fill_gap_max_hops: bpy.props.IntProperty(
        name='Max Gap Hops',
        default=2,
        min=1,
        max=3,
        description='Maximum editable mesh edge hops used to fill small seam gaps.',
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
