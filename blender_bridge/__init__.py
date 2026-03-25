from __future__ import annotations

bl_info = {
    'name': 'UV Seam Predictor',
    'author': 'UV Seam GNN Pipeline',
    'version': (0, 5, 0),
    'blender': (4, 0, 0),
    'location': 'View3D › N-panel › UV Seam GNN',
    'description': 'Auto-mark UV seams via an external GNN inference process.',
    'category': 'UV',
}

import os
import subprocess
import sys
import tempfile

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, StringProperty
from bpy.types import Operator, Panel

_ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
_INFERENCE_PY = os.path.join(_ADDON_DIR, 'run_inference.py')


def _mesh_to_arrays(obj) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract raw mesh geometry from a Blender mesh without importing torch.

    Returns:
      vertices     [N, 3] float32  vertex positions
      normals      [N, 3] float32  vertex normals
      faces        [F, 3] int64    triangulated face vertex indices
      unique_edges [E, 2] int64    undirected edges (vi < vj)
    """
    import bmesh

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    vertices = np.array(
        [[v.co.x, v.co.y, v.co.z] for v in bm.verts], dtype=np.float32
    )
    normals = np.array(
        [[v.normal.x, v.normal.y, v.normal.z] for v in bm.verts], dtype=np.float32
    )
    faces = np.array(
        [[f.verts[0].index, f.verts[1].index, f.verts[2].index] for f in bm.faces],
        dtype=np.int64,
    )
    unique_edges = np.array(
        [(min(e.verts[0].index, e.verts[1].index),
          max(e.verts[0].index, e.verts[1].index))
         for e in bm.edges],
        dtype=np.int64,
    )

    bm.free()
    eval_obj.to_mesh_clear()

    return vertices, normals, faces, unique_edges


class UVSeamGNNProperties(bpy.types.PropertyGroup):
    python_exe: StringProperty(
        name='Python Exe',
        description=(
            'Python executable with torch + torch_geometric installed. '
            'Use an absolute path if "python" is ambiguous '
            '(e.g. C:/Users/you/.venv/Scripts/python.exe)'
        ),
        default='python',
    )
    weights_path: StringProperty(
        name='Weights',
        description='Path to best_model.pth',
        default='',
        subtype='FILE_PATH',
    )
    model_type: EnumProperty(
        name='Model',
        description='GNN architecture of the loaded weights',
        items=[
            ('graphsage', 'DualGraphSAGE', 'GraphSAGE on the dual graph (default)'),
            ('gatv2', 'DualGATv2', 'GATv2 with multi-head attention on the dual graph'),
            ('meshcnn', 'MeshCNN', 'MeshCNN edge convolution on original mesh topology'),
        ],
        default='graphsage',
    )
    threshold: FloatProperty(
        name='Threshold',
        description='Sigmoid threshold for seam prediction (0–1)',
        default=0.5,
        min=0.0,
        max=1.0,
    )


class OBJECT_OT_test_uv_seam_python(Operator):
    bl_idname = 'object.test_uv_seam_python'
    bl_label = 'Test Python'
    bl_description = 'Check that the configured Python can import torch and torch_geometric'

    def execute(self, context):
        python = context.scene.uv_seam_gnn.python_exe
        check = (
            'import torch, torch_geometric; '
            'print(f"torch {torch.__version__}, '
            'torch_geometric {torch_geometric.__version__}")'
        )
        result = subprocess.run([python, '-c', check], capture_output=True, text=True)
        if result.returncode == 0:
            self.report({'INFO'}, f"OK — {result.stdout.strip()}")
        else:
            err = (result.stderr or result.stdout).strip()
            self.report({'ERROR'}, f"Failed: {err[:200]}")
        return {'FINISHED'}


class OBJECT_OT_predict_uv_seams(Operator):
    bl_idname = 'object.predict_uv_seams'
    bl_label = 'Auto-Mark UV Seams'
    bl_description = 'Run GNN inference via subprocess and mark predicted seam edges'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.uv_seam_gnn

        weights_path = bpy.path.abspath(props.weights_path)
        if not props.weights_path:
            self.report({'ERROR'}, 'Set the model weights path first.')
            return {'CANCELLED'}
        if not os.path.isfile(weights_path):
            self.report({'ERROR'}, f'Weights file not found: {weights_path}')
            return {'CANCELLED'}

        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, 'Select a mesh object.')
            return {'CANCELLED'}

        # extract raw mesh geometry (numpy only — no torch needed here)
        try:
            vertices, normals, faces, unique_edges = _mesh_to_arrays(obj)
        except Exception as exc:
            self.report({'ERROR'}, f'Mesh extraction failed: {exc}')
            return {'CANCELLED'}

        # communicate with the inference subprocess via temp files
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, 'mesh.npz')
            output_path = os.path.join(tmp, 'seams.txt')

            np.savez(
                data_path,
                vertices=vertices,
                normals=normals,
                faces=faces,
                unique_edges=unique_edges,
            )

            cmd = [
                props.python_exe,
                _INFERENCE_PY,
                data_path,
                weights_path,
                str(props.threshold),
                output_path,
                '--model-type', props.model_type,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                err = (result.stderr or result.stdout).strip()
                self.report({'ERROR'}, f'Inference failed: {err[:300]}')
                print(f'[uv seam gnn] subprocess stderr:\n{result.stderr}')
                return {'CANCELLED'}

            if not os.path.isfile(output_path):
                self.report({'ERROR'}, 'Inference script produced no output.')
                return {'CANCELLED'}

            with open(output_path) as f:
                raw = f.read().strip()

            seam_indices = (
                [int(x) for x in raw.splitlines() if x.strip()]
                if raw else []
            )

        mesh = obj.data
        for edge in mesh.edges:
            edge.use_seam = False

        edge_key_to_idx = {
            (min(e.vertices[0], e.vertices[1]),
             max(e.vertices[0], e.vertices[1])): i
            for i, e in enumerate(mesh.edges)
        }

        seam_count = 0
        for k in seam_indices:
            if k >= len(unique_edges):
                continue
            vi, vj = int(unique_edges[k, 0]), int(unique_edges[k, 1])
            key = (min(vi, vj), max(vi, vj))
            if (idx := edge_key_to_idx.get(key)) is not None:
                mesh.edges[idx].use_seam = True
                seam_count += 1

        self.report({'INFO'}, f"marked {seam_count} seam edge(s) on '{obj.name}'.")
        return {'FINISHED'}


class VIEW3D_PT_uv_seam_gnn(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'UV Seam GNN'
    bl_label = 'UV Seam Predictor'

    def draw(self, context):
        layout = self.layout
        props = context.scene.uv_seam_gnn

        box = layout.box()
        box.label(text='Python (must have torch + pyg):', icon='CONSOLE')
        box.prop(props, 'python_exe', text='')
        box.operator('object.test_uv_seam_python', icon='CHECKMARK')

        layout.separator()
        layout.prop(props, 'weights_path')
        layout.prop(props, 'model_type')
        layout.prop(props, 'threshold', slider=True)
        layout.separator()
        layout.operator('object.predict_uv_seams', icon='UV')


_CLASSES = [
    UVSeamGNNProperties,
    OBJECT_OT_test_uv_seam_python,
    OBJECT_OT_predict_uv_seams,
    VIEW3D_PT_uv_seam_gnn,
]


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.uv_seam_gnn = bpy.props.PointerProperty(type=UVSeamGNNProperties)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.uv_seam_gnn


if __name__ == '__main__':
    register()
