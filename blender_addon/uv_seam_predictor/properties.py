import bpy


class UVSEAM_Settings(bpy.types.PropertyGroup):
    model_weights_path: bpy.props.StringProperty(
        name='Model Weights Path',
        subtype='FILE_PATH',
        description='Path to the trained model weights file',
    )
    threshold: bpy.props.FloatProperty(
        name='Inference Seam Threshold',
        default=0.75,
        min=0.0,
        max=1.0,
        description='Probability threshold for raw seam classification before post-processing.',
    )
    use_post_processing: bpy.props.BoolProperty(
        name='Apply Post-processing',
        default=True,
        description='Run inference-time skeletonization, endpoint bridging, spur pruning, '
                    'and Blender editable-mesh small-gap filling.',
    )
    postprocess_tau_low: bpy.props.FloatProperty(
        name='Skeleton Candidate Threshold',
        default=0.30,
        min=0.0,
        max=1.0,
        description='Lower probability threshold used to choose seam candidates for '
                    'inference-time skeletonization and thinning.',
    )
    postprocess_tau_high: bpy.props.FloatProperty(
        name='Legacy Tau High (debug)',
        default=0.70,
        min=0.0,
        max=1.0,
        description='Legacy compatibility/debug value kept for saved settings. '
                    'It does not affect current endpoint-bridging behavior.',
    )
    postprocess_d_max: bpy.props.IntProperty(
        name='Skeleton Thickness Distance',
        default=3,
        min=1,
        max=16,
        description='Maximum mesh-edge distance allowed between thinned seam candidates '
                    'and the preserved inference-time skeleton.',
    )
    postprocess_r_bridge: bpy.props.IntProperty(
        name='Endpoint Bridge Max Hops',
        default=6,
        min=0,
        max=32,
        description='Maximum mesh-edge hops for inference-time endpoint bridging. '
                    '0 disables endpoint bridging.',
    )
    postprocess_l_min: bpy.props.IntProperty(
        name='Spur Prune Min Branch Length',
        default=4,
        min=1,
        max=32,
        description='Minimum branch length kept during inference-time spur pruning '
                    'and small dangling-branch cleanup.',
    )
    postprocess_epsilon: bpy.props.FloatProperty(
        name='Legacy Epsilon (debug)',
        default=1e-3,
        min=1e-6,
        max=0.1,
        precision=6,
        description='Legacy compatibility/debug numerical floor kept for saved settings. '
                    'It does not affect current endpoint-bridging behavior.',
    )
    postprocess_anchor_boundary: bpy.props.BoolProperty(
        name='Anchor Boundary for Skeleton Cleanup',
        default=True,
        description='Treat mesh boundary vertices as anchors for inference-time '
                    'skeletonization and spur-pruning cleanup.',
    )
    postprocess_fill_small_gaps: bpy.props.BoolProperty(
        name='Fill Blender Mesh Small Gaps',
        default=True,
        description='Fill bounded small seam gaps on the editable Blender mesh after prediction.',
    )
    postprocess_fill_gap_max_hops: bpy.props.IntProperty(
        name='Editable Gap Max Hops',
        default=2,
        min=1,
        soft_max=3,
        description='Maximum editable mesh edge hops for Blender-side gap filling. '
                    '3 is recommended; higher values may over-connect seams.',
    )
    postprocess_write_debug_sidecars: bpy.props.BoolProperty(
        name='Write Legacy Debug Sidecars',
        default=False,
        description='Debug only: write legacy post-processing diagnostic JSON sidecars.',
    )
    manual_cleanup_max_dangling_edges: bpy.props.IntProperty(
        name='Max Dangling Length',
        default=1,
        min=1,
        soft_max=3,
        description='Maximum seam-edge length of dangling branches removed by manual cleanup. '
                    '1 is safest.',
    )
    manual_cleanup_protect_boundary_vertices: bpy.props.BoolProperty(
        name='Protect Boundary Ends',
        default=True,
        description='Do not remove dangling branches anchored at mesh boundary vertices.',
    )
    manual_mirror_axis: bpy.props.EnumProperty(
        name='Mirror Axis',
        items=(
            ('X', 'X', 'Local X axis'),
            ('Y', 'Y', 'Local Y axis'),
            ('Z', 'Z', 'Local Z axis'),
        ),
        default='X',
        description='Local mesh axis used for manual seam mirroring.',
    )
    manual_mirror_tolerance: bpy.props.FloatProperty(
        name='Mirror Tolerance',
        default=1e-4,
        min=1e-8,
        soft_max=1e-2,
        precision=6,
        description='Coordinate tolerance used to match mirrored vertices in object-local space.',
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
