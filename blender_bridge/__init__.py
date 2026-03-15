from __future__ import annotations

bl_info = {
    'name': 'UV Seam Predictor',
    'author': 'UV Seam GNN Pipeline',
    'version': (0, 4, 0),
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
from bpy.props import FloatProperty, StringProperty
from bpy.types import Operator, Panel

_ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
_INFERENCE_PY = os.path.join(_ADDON_DIR, 'run_inference.py')


def _mesh_to_arrays(obj) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Extract graph arrays from a Blender mesh without importing torch.

    Returns (x, edge_index, edge_attr, unique_edges) where:
      x            [N, 6]   vertex positions + normals
      edge_index   [2, 2*E] bidirectional
      edge_attr    [2*E, 4] [length, dihedral≈π, delta_normal, dot_normal]
      unique_edges           list of (vi, vj) pairs (length E)
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

    verts = np.array(
        [[v.co.x, v.co.y, v.co.z, v.normal.x, v.normal.y, v.normal.z]
         for v in bm.verts],
        dtype=np.float32,
    )

    unique_edges = [(e.verts[0].index, e.verts[1].index) for e in bm.edges]
    vi_arr = np.array([vi for vi, _ in unique_edges], dtype=np.int64)
    vj_arr = np.array([vj for _, vj in unique_edges], dtype=np.int64)

    edge_index = np.array(
        [np.concatenate([vi_arr, vj_arr]),
         np.concatenate([vj_arr, vi_arr])],
        dtype=np.int64,
    )

    edge_vec = verts[vj_arr, :3] - verts[vi_arr, :3]
    edge_len = np.linalg.norm(edge_vec, axis=1, keepdims=True).astype(np.float32)
    dihedral = np.full((len(unique_edges), 1), np.pi, dtype=np.float32)

    ni, nj = verts[vi_arr, 3:], verts[vj_arr, 3:]
    delta_nrm = np.linalg.norm(ni - nj, axis=1, keepdims=True).astype(np.float32)

    eps = 1e-8
    ni_hat = ni / np.linalg.norm(ni, axis=1, keepdims=True).clip(min=eps)
    nj_hat = nj / np.linalg.norm(nj, axis=1, keepdims=True).clip(min=eps)
    dot_nrm = np.einsum('ij,ij->i', ni_hat, nj_hat).astype(np.float32)[:, None]

    edge_attr = np.tile(
        np.concatenate([edge_len, dihedral, delta_nrm, dot_nrm], axis=1),
        (2, 1),
    )

    bm.free()
    eval_obj.to_mesh_clear()

    return verts, edge_index, edge_attr, unique_edges


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

        # build graph arrays inside Blender (numpy only — no torch needed here)
        try:
            verts, edge_index, edge_attr, unique_edges = _mesh_to_arrays(obj)
        except Exception as exc:
            self.report({'ERROR'}, f'Graph build failed: {exc}')
            return {'CANCELLED'}

        # communicate with the inference subprocess via temp files
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, 'mesh.npz')
            output_path = os.path.join(tmp, 'seams.txt')

            np.savez(
                data_path,
                x=verts,
                edge_index=edge_index,
                edge_attr=edge_attr,
                n_unique=np.array(len(unique_edges)),
            )

            cmd = [
                props.python_exe,
                _INFERENCE_PY,
                data_path,
                weights_path,
                str(props.threshold),
                output_path,
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
            vi, vj = unique_edges[k]
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
