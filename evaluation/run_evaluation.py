"""
UV-level evaluation orchestrator.

For each test mesh, runs GNN inference, Blender unwrap (predicted seams),
Smart UV Project baseline, and compares UV quality against the ground truth
UV already embedded in the .obj file.

Usage:
    python evaluation/run_evaluation.py \\
        --test-meshes ./3d-objs/ \\
        --dual-dataset dataset_dual.pt \\
        --weights runs/dual_graphsage_001/best_model.pth \\
        --model-type graphsage \\
        --blender-exe blender \\
        --output-dir evaluation/results/graphsage_eval \\
        [--threshold 0.5] \\
        [--min-component 3] \\
        [--max-gap 3] \\
        [--unwrap-method ANGLE_BASED] \\
        [--max-meshes N]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# allow imports from project root
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from preprocessing.compute_features import compute_edge_features
from preprocessing.build_dual_graph import build_dual_graph_data
from models.utils.postprocess import threshold_and_clean, stitch_seam_gaps
from evaluation.uv_metrics import parse_obj_with_uv, compute_all_uv_metrics

import trimesh

# plot style constants (match experiment_log.py)
_C_OURS = '#2196F3'
_C_GT = '#4CAF50'
_C_SMART = '#FF5722'
_DPI = 150


# ─── Inference ───────────────────────────────────────────────────────────────

def _load_model(weights_path: str, model_type: str, device: torch.device):
    if model_type == 'graphsage':
        from models.dual_graphsage.model import DualGraphSAGE
        model = DualGraphSAGE().to(device)
    elif model_type == 'gatv2':
        from models.gatv2.model import DualGATv2
        model = DualGATv2().to(device)
    elif model_type == 'meshcnn':
        from models.meshcnn.model import MeshCNNClassifier
        model = MeshCNNClassifier().to(device)
    else:
        raise ValueError(f'Unknown model type: {model_type!r}')

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _infer_seam_indices(
    mesh_path: str,
    model,
    model_type: str,
    device: torch.device,
    threshold: float,
    min_component: int,
    max_gap: int,
) -> tuple[list[int], np.ndarray]:
    """Run inference on a single .obj and return post-processed seam edge indices."""
    mesh = trimesh.load(str(mesh_path), process=False, force='mesh')
    features, unique_edges, _ = compute_edge_features(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if model_type == 'meshcnn':
        from preprocessing.build_meshcnn_data import build_edge_neighbors
        src = unique_edges[:, 0]
        dst = unique_edges[:, 1]
        edge_key_to_idx = {
            (int(min(src[i], dst[i])), int(max(src[i], dst[i]))): i
            for i in range(len(unique_edges))
        }
        neighbors = build_edge_neighbors(src, dst, faces, edge_key_to_idx)
        x = torch.from_numpy(features).float().to(device)
        nb = torch.from_numpy(neighbors).long().to(device)
        with torch.no_grad():
            logits = model(x, nb)
    else:
        # dual graph path (graphsage / gatv2)
        from torch_geometric.data import Data
        n_verts = len(mesh.vertices)
        edge_index_fwd = torch.from_numpy(
            np.stack([unique_edges[:, 0], unique_edges[:, 1]], axis=0)
        ).long()
        edge_index_bwd = torch.from_numpy(
            np.stack([unique_edges[:, 1], unique_edges[:, 0]], axis=0)
        ).long()
        full_edge_index = torch.cat([edge_index_fwd, edge_index_bwd], dim=1)
        data = Data(
            x=torch.zeros(n_verts, 6),
            edge_index=full_edge_index,
            edge_attr=torch.from_numpy(features).float(),
            y=torch.zeros(len(unique_edges) * 2),
            faces=torch.from_numpy(faces),
        )
        data.file_path = str(mesh_path)
        dual_data = build_dual_graph_data(data)
        x = dual_data.x.to(device)
        ei = dual_data.edge_index.to(device)
        with torch.no_grad():
            logits = model(x, ei)

    probs = torch.sigmoid(logits).cpu().numpy()

    # build edge_to_faces for gap stitching
    edge_to_faces: dict = {}
    for f_idx, face in enumerate(faces):
        for k in range(3):
            vi, vj = int(face[k]), int(face[(k + 1) % 3])
            key = (min(vi, vj), max(vi, vj))
            edge_to_faces.setdefault(key, []).append(f_idx)

    mask = threshold_and_clean(probs, unique_edges, threshold, min_component)
    mask = stitch_seam_gaps(probs, mask, unique_edges, edge_to_faces, max_gap)

    return np.where(mask)[0].tolist(), unique_edges


def _edge_metrics(
    mesh_path: str,
    seam_indices: list[int],
    unique_edges: np.ndarray,
) -> dict:
    """Compute edge-level F1 by comparing predicted seams against .obj UV seams."""
    try:
        data = parse_obj_with_uv(mesh_path)
        if data['uv_faces'] is None:
            return {}

        verts = data['vertices']
        faces = data['faces']
        uv_faces = data['uv_faces']

        # build ground truth seam mask: edges with different UV indices on two sides
        from evaluation.uv_metrics import seam_length as _sl  # just to reuse mesh edge logic
        edge_key_to_idx = {
            (int(unique_edges[i, 0]), int(unique_edges[i, 1])): i
            for i in range(len(unique_edges))
        }
        mesh_edge_to_uvs: dict = {}
        for fi, (face, uv_tri) in enumerate(zip(faces, uv_faces)):
            for k in range(3):
                vi, vj = int(face[k]), int(face[(k + 1) % 3])
                ui, uj = int(uv_tri[k]), int(uv_tri[(k + 1) % 3])
                key = (min(vi, vj), max(vi, vj))
                mesh_edge_to_uvs.setdefault(key, []).append((ui, uj))

        gt_seam = np.zeros(len(unique_edges), dtype=bool)
        for key, uv_pairs in mesh_edge_to_uvs.items():
            if len(uv_pairs) < 2:
                gt_seam[edge_key_to_idx.get(key, -1)] = True  # boundary = seam
                continue
            (ui0, uj0), (ui1, uj1) = uv_pairs[0], uv_pairs[1]
            same = (ui0 == ui1 and uj0 == uj1) or (ui0 == uj1 and uj0 == ui1)
            if not same:
                idx = edge_key_to_idx.get(key)
                if idx is not None:
                    gt_seam[idx] = True

        pred_seam = np.zeros(len(unique_edges), dtype=bool)
        for i in seam_indices:
            if 0 <= i < len(unique_edges):
                pred_seam[i] = True

        tp = int(np.sum(pred_seam & gt_seam))
        fp = int(np.sum(pred_seam & ~gt_seam))
        fn = int(np.sum(~pred_seam & gt_seam))
        tn = int(np.sum(~pred_seam & ~gt_seam))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / len(unique_edges) if len(unique_edges) > 0 else 0.0

        return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': acc}
    except Exception as e:
        print(f'  [warn] edge metrics failed: {e}')
        return {}


# ─── Blender subprocess ──────────────────────────────────────────────────────

_BLENDER_SCRIPT = str(Path(__file__).parent / 'blender_unwrap.py')


def _run_blender(blender_exe: str, args: list[str]) -> bool:
    cmd = [blender_exe, '-b', '--factory-startup', '-P', _BLENDER_SCRIPT, '--'] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  [warn] blender failed:\n{result.stderr[-500:]}')
        return False
    return True


# ─── Plots ───────────────────────────────────────────────────────────────────

def _plot_comparison_table(summary: dict, output_path: str) -> None:
    import matplotlib.pyplot as plt

    methods = ['ground_truth', 'predicted', 'smart_uv']
    labels = ['Ground Truth', 'Ours', 'Smart UV']
    metric_names = [
        'area_distortion_avg', 'area_distortion_max',
        'angle_distortion_avg', 'angle_distortion_max',
        'symmetric_dirichlet_avg', 'flipped_pct',
        'num_shells', 'seam_length',
    ]
    display_names = [
        'Area Distortion (avg)', 'Area Distortion (max)',
        'Angle Distortion (avg)', 'Angle Distortion (max)',
        'Sym. Dirichlet (avg)', 'Flipped Triangles (%)',
        'UV Shells', 'Seam Length',
    ]

    agg = summary.get('aggregated', {})
    cell_data = []
    col_colors = []

    for metric, display in zip(metric_names, display_names):
        row = [display]
        row_colors = ['#F5F5F5']
        vals = []
        for method in methods:
            v = agg.get(method, {}).get(metric, {}).get('mean', float('nan'))
            vals.append(v)

        # bold (darker bg) the best value in each row, excluding GT
        non_gt_vals = vals[1:]  # ours + smart_uv
        best_idx = None
        if not all(np.isnan(v) for v in non_gt_vals):
            # lower is better for all metrics except num_shells (neutral)
            best_idx = 1 + int(np.nanargmin(non_gt_vals))

        for col_idx, v in enumerate(vals):
            cell = f'{v:.4f}' if not np.isnan(v) else 'N/A'
            row.append(cell)
            color = '#BBDEFB' if (col_idx + 1 == best_idx) else 'white'
            row_colors.append(color)

        cell_data.append(row)
        col_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(metric_names) + 1.5))
    ax.axis('off')
    tbl = ax.table(
        cellText=cell_data,
        colLabels=['Metric'] + labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(labels) + 1)))

    # apply cell colors
    for row_idx, row_colors in enumerate(col_colors):
        for col_idx, color in enumerate(row_colors):
            tbl[row_idx + 1, col_idx].set_facecolor(color)

    # header color
    for col_idx in range(len(labels) + 1):
        tbl[0, col_idx].set_facecolor('#1565C0')
        tbl[0, col_idx].set_text_props(color='white', fontweight='bold')

    plt.title('UV Quality Comparison', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI, bbox_inches='tight')
    plt.close()


def _plot_distortion_bars(summary: dict, output_path: str) -> None:
    import matplotlib.pyplot as plt

    metrics = ['area_distortion_avg', 'angle_distortion_avg', 'symmetric_dirichlet_avg']
    display = ['Area Distortion', 'Angle Distortion', 'Sym. Dirichlet']
    methods = ['ground_truth', 'predicted', 'smart_uv']
    colors = [_C_GT, _C_OURS, _C_SMART]
    method_labels = ['Ground Truth', 'Ours', 'Smart UV']

    agg = summary.get('aggregated', {})
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (method, color, label) in enumerate(zip(methods, colors, method_labels)):
        vals = [agg.get(method, {}).get(m, {}).get('mean', 0.0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(display)
    ax.set_ylabel('Distortion')
    ax.set_title('UV Distortion Comparison', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI)
    plt.close()


def _plot_shells_comparison(summary: dict, output_path: str) -> None:
    import matplotlib.pyplot as plt

    methods = ['ground_truth', 'predicted', 'smart_uv']
    labels = ['Ground Truth', 'Ours', 'Smart UV']
    colors = [_C_GT, _C_OURS, _C_SMART]
    agg = summary.get('aggregated', {})

    vals = [agg.get(m, {}).get('num_shells', {}).get('mean', 0.0) for m in methods]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_ylabel('Avg. UV Shells (Islands)')
    ax.set_title('UV Shell Count Comparison', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI)
    plt.close()


def _plot_per_mesh_scatter(per_mesh: list[dict], output_path: str) -> None:
    import matplotlib.pyplot as plt

    gt_vals = [r.get('ground_truth', {}).get('area_distortion_avg', float('nan'))
               for r in per_mesh]
    pred_vals = [r.get('predicted', {}).get('area_distortion_avg', float('nan'))
                 for r in per_mesh]

    gt_vals = np.array(gt_vals, dtype=float)
    pred_vals = np.array(pred_vals, dtype=float)
    valid = ~(np.isnan(gt_vals) | np.isnan(pred_vals))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt_vals[valid], pred_vals[valid], color=_C_OURS, alpha=0.7, s=50)
    lim = max(gt_vals[valid].max(), pred_vals[valid].max()) * 1.05 if valid.any() else 1.0
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y = x (perfect match)')
    ax.set_xlabel('Ground Truth Area Distortion')
    ax.set_ylabel('Predicted Area Distortion')
    ax.set_title('Per-Mesh Distortion Correlation', fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI)
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='UV-level evaluation pipeline.')
    parser.add_argument('--test-meshes', required=True,
                        help='Directory of .obj test meshes')
    parser.add_argument('--dual-dataset', default='dataset_dual.pt',
                        help='Path to dual graph dataset (for reference, not loaded here)')
    parser.add_argument('--weights', required=True, help='Path to best_model.pth')
    parser.add_argument('--model-type', required=True, choices=['graphsage', 'gatv2', 'meshcnn'])
    parser.add_argument('--blender-exe', default='blender',
                        help='Blender executable path (default: blender)')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write results')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-component', type=int, default=3)
    parser.add_argument('--max-gap', type=int, default=3)
    parser.add_argument('--unwrap-method', default='ANGLE_BASED',
                        choices=['ANGLE_BASED', 'CONFORMAL'])
    parser.add_argument('--max-meshes', type=int, default=None,
                        help='Limit number of meshes evaluated (for testing)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # find test meshes (skip augmented copies)
    mesh_dir = Path(args.test_meshes)
    mesh_paths = sorted([
        p for p in mesh_dir.glob('*.obj')
        if '_aug' not in p.stem
    ])
    if args.max_meshes:
        mesh_paths = mesh_paths[:args.max_meshes]

    if not mesh_paths:
        print(f'[eval] no .obj files found in {mesh_dir}')
        sys.exit(1)

    print(f'[eval] evaluating {len(mesh_paths)} mesh(es) → {out_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _load_model(args.weights, args.model_type, device)

    per_mesh_results = []

    for mesh_path in mesh_paths:
        name = mesh_path.name
        print(f'\n[eval] {name}')

        record: dict = {'mesh': name}

        # Ground truth UV is already in the .obj
        gt_data = parse_obj_with_uv(str(mesh_path))
        if gt_data['uv_coords'] is None:
            print(f'  [skip] no UV in {name}')
            continue
        gt_metrics = compute_all_uv_metrics(
            gt_data['vertices'], gt_data['faces'],
            gt_data['uv_coords'], gt_data['uv_faces'],
        )
        record['ground_truth'] = gt_metrics

        with tempfile.TemporaryDirectory() as tmp:
            seams_file = os.path.join(tmp, 'seams.txt')
            pred_obj = os.path.join(tmp, 'pred_unwrap.obj')
            smart_obj = os.path.join(tmp, 'smart_unwrap.obj')

            # ── Inference ────────────────────────────────────────────────────
            try:
                seam_indices, unique_edges = _infer_seam_indices(
                    str(mesh_path), model, args.model_type, device,
                    args.threshold, args.min_component, args.max_gap,
                )
                print(f'  inference: {len(seam_indices)} seam edges predicted')
            except Exception as e:
                print(f'  [warn] inference failed: {e}')
                seam_indices, unique_edges = [], np.zeros((0, 2), dtype=np.int64)

            with open(seams_file, 'w') as sf:
                sf.write('\n'.join(map(str, seam_indices)))

            # edge-level metrics
            if len(unique_edges) > 0:
                em = _edge_metrics(str(mesh_path), seam_indices, unique_edges)
                if em:
                    record['edge_metrics'] = em
                    print(f'  edge F1={em.get("f1", 0):.3f}  '
                          f'P={em.get("precision", 0):.3f}  '
                          f'R={em.get("recall", 0):.3f}')

            # ── Predicted seam unwrap (Blender) ───────────────────────────────
            ok = _run_blender(args.blender_exe, [
                '--input', str(mesh_path),
                '--seams', seams_file,
                '--output', pred_obj,
                '--method', args.unwrap_method,
            ])
            if ok and os.path.isfile(pred_obj):
                pred_data = parse_obj_with_uv(pred_obj)
                record['predicted'] = compute_all_uv_metrics(
                    pred_data['vertices'], pred_data['faces'],
                    pred_data['uv_coords'], pred_data['uv_faces'],
                )
                print(f'  predicted unwrap: '
                      f'area_d={record["predicted"].get("area_distortion_avg", 0):.4f}  '
                      f'shells={record["predicted"].get("num_shells", 0)}')
            else:
                record['predicted'] = None

            # ── Smart UV Project (Blender) ─────────────────────────────────────
            ok = _run_blender(args.blender_exe, [
                '--input', str(mesh_path),
                '--output', smart_obj,
                '--smart-uv',
            ])
            if ok and os.path.isfile(smart_obj):
                smart_data = parse_obj_with_uv(smart_obj)
                record['smart_uv'] = compute_all_uv_metrics(
                    smart_data['vertices'], smart_data['faces'],
                    smart_data['uv_coords'], smart_data['uv_faces'],
                )
                print(f'  smart UV:         '
                      f'area_d={record["smart_uv"].get("area_distortion_avg", 0):.4f}  '
                      f'shells={record["smart_uv"].get("num_shells", 0)}')
            else:
                record['smart_uv'] = None

        per_mesh_results.append(record)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    metric_keys = [
        'area_distortion_avg', 'area_distortion_max',
        'angle_distortion_avg', 'angle_distortion_max',
        'symmetric_dirichlet_avg', 'flipped_pct', 'num_shells', 'seam_length',
    ]
    edge_keys = ['f1', 'precision', 'recall', 'accuracy']

    def _agg(records: list, method_key: str, keys: list) -> dict:
        result = {}
        for k in keys:
            vals = [r[method_key][k] for r in records
                    if r.get(method_key) and k in r[method_key] and not np.isnan(r[method_key][k])]
            result[k] = {
                'mean': float(np.mean(vals)) if vals else float('nan'),
                'std': float(np.std(vals)) if vals else float('nan'),
            }
        return result

    edge_records = [r for r in per_mesh_results if r.get('edge_metrics')]

    summary = {
        'model': args.model_type,
        'weights': args.weights,
        'n_meshes': len(per_mesh_results),
        'threshold': args.threshold,
        'unwrap_method': args.unwrap_method,
        'aggregated': {
            'ground_truth': _agg(per_mesh_results, 'ground_truth', metric_keys),
            'predicted': _agg(
                [r for r in per_mesh_results if r.get('predicted')],
                'predicted', metric_keys,
            ),
            'smart_uv': _agg(
                [r for r in per_mesh_results if r.get('smart_uv')],
                'smart_uv', metric_keys,
            ),
            'edge_metrics': _agg(edge_records, 'edge_metrics', edge_keys),
        },
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(out_dir / 'per_mesh_results.json', 'w') as f:
        json.dump(per_mesh_results, f, indent=2, default=str)

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f'\n[eval] results saved to {out_dir}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        _plot_comparison_table(summary, str(out_dir / 'comparison_table.png'))
        _plot_distortion_bars(summary, str(out_dir / 'distortion_bars.png'))
        _plot_shells_comparison(summary, str(out_dir / 'shells_comparison.png'))
        _plot_per_mesh_scatter(per_mesh_results, str(out_dir / 'per_mesh_scatter.png'))
        print(f'[eval] plots saved to {out_dir}')
    except Exception as e:
        print(f'[eval] plot generation failed: {e}')

    # ── Print summary ─────────────────────────────────────────────────────────
    agg = summary['aggregated']
    print(f'\n{"─"*60}')
    print(f'{"Metric":<30s}  {"GT":>8s}  {"Ours":>8s}  {"SmartUV":>8s}')
    print(f'{"─"*60}')
    for k in metric_keys:
        gt_v = agg['ground_truth'].get(k, {}).get('mean', float('nan'))
        p_v = agg['predicted'].get(k, {}).get('mean', float('nan'))
        s_v = agg['smart_uv'].get(k, {}).get('mean', float('nan'))
        print(f'{k:<30s}  {gt_v:>8.4f}  {p_v:>8.4f}  {s_v:>8.4f}')

    if edge_keys:
        print(f'\n{"─"*60}')
        print('Edge-level metrics (predicted vs GT seams):')
        for k in edge_keys:
            v = agg['edge_metrics'].get(k, {}).get('mean', float('nan'))
            print(f'  {k:<12s}: {v:.4f}')


if __name__ == '__main__':
    main()
