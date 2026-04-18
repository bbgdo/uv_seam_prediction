SMOKE_CHECKLIST = {
    'registration_ui': (
        'Add-on registers cleanly.',
        'Panel appears in the View3D sidebar.',
        'Panel and category say "Auto Seams".',
        'Model weights path is visible in the N-panel.',
        'Feature bundle selector is absent.',
        'Threshold slider is visible exactly once.',
        (
            'Add-on Preferences only contain python executable, predict script path, '
            'timeout, temp file, and error log settings; model weights path is absent.'
        ),
    ),
    'basic_validation': (
        'Active non-mesh object is rejected cleanly.',
        'Missing python executable path is rejected cleanly.',
        'Missing predict script path is rejected cleanly.',
        'Missing model weights path is rejected cleanly.',
        'Enabled modifiers are rejected cleanly.',
        'Shared mesh datablock behavior is safe with Make Mesh Single User on and off.',
    ),
    'hidden_triangulation_export': (
        'Quad mesh is accepted without asking the artist to triangulate.',
        'N-gon mesh is accepted without asking the artist to triangulate.',
        'Export path triangulates only a temporary bmesh copy.',
        'Original mesh polygon counts are unchanged after export.',
        'Original edge counts are unchanged after export.',
        'Original vertex counts are unchanged after export.',
        'OBJ vertex numbering still matches original Blender vertex indices.',
        'Triangulated export produces triangle faces only.',
    ),
    'mapping_safety': (
        'Predicted seam on an original quad boundary edge is applied.',
        'Predicted seam on an original n-gon boundary edge is applied.',
        'Predicted seam on a triangulation-only diagonal is ignored.',
        'Ignored triangulation-only diagonals do not raise a hard failure.',
        'Duplicates are skipped and counted cleanly.',
        'Summary text clearly distinguishes applied seams from ignored triangulation-only diagonals.',
    ),
    'modal_execution_safety': (
        'Operator enters running state correctly.',
        'Timer is removed on success.',
        'Timer is removed on failure.',
        'Original mode is restored on success.',
        'Original mode is restored on failure.',
        'Temp files are cleaned up according to settings.',
        'Topology-change guard prevents stale prediction application if topology changes during runtime.',
    ),
    'architecture_agnosticism': (
        'Add-on does not pass --feature-bundle anymore.',
        'Add-on only passes model weights path and threshold to predict_seams.py.',
        'Add-on does not branch on model architecture type internally.',
    ),
}


def iter_smoke_checklist():
    for section, items in SMOKE_CHECKLIST.items():
        yield section, items


def print_smoke_checklist():
    for section, items in iter_smoke_checklist():
        print(section)
        for index, item in enumerate(items, start=1):
            print(f'  {index}. {item}')
