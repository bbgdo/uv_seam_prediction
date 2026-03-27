import argparse
import json
import sys
from pathlib import Path

import numpy as np

_DPI = 150

# Colors: one per model, GT and Smart UV get fixed colors
_COLORS_MODEL = ['#2196F3', '#FF9800', '#9C27B0', '#00BCD4']
_C_GT = '#4CAF50'
_C_SMART = '#FF5722'


def _load_summary(result_dir: str) -> dict:
    path = Path(result_dir) / 'summary.json'
    if not path.exists():
        print(f'[compare] warning: {path} not found — skipping')
        return None
    with open(path) as f:
        return json.load(f)


def _model_label(summary: dict, result_dir: str) -> str:
    model = summary.get('model', '')
    if model == 'graphsage':
        return 'DualGraphSAGE'
    if model == 'gatv2':
        return 'DualGATv2'
    return Path(result_dir).name


def _get_mean(summary: dict, method: str, metric: str) -> float:
    return summary.get('aggregated', {}).get(method, {}).get(metric, {}).get('mean', float('nan'))


def plot_full_comparison_table(summaries: list[dict], labels: list[str], output_path: str) -> None:
    import matplotlib.pyplot as plt

    metric_keys = [
        'area_distortion_avg', 'area_distortion_max',
        'angle_distortion_avg', 'angle_distortion_max',
        'symmetric_dirichlet_avg', 'flipped_pct',
        'num_shells', 'seam_length',
    ]
    edge_keys = ['f1', 'precision', 'recall']

    display_metrics = [
        'Area Dist. (avg)', 'Area Dist. (max)',
        'Angle Dist. (avg)', 'Angle Dist. (max)',
        'Sym. Dirichlet', 'Flipped (%)',
        'UV Shells', 'Seam Length',
        'F1', 'Precision', 'Recall',
    ]

    # Build rows: one per model's predicted results, then Smart UV, then GT
    row_labels = labels + ['Smart UV Project', 'Ground Truth']
    n_rows = len(row_labels)
    n_cols = len(display_metrics)

    cell_data = []
    raw_vals = []  # [n_rows, n_metrics] for best-value detection

    for label, summary in zip(labels, summaries):
        row = []
        for k in metric_keys:
            v = _get_mean(summary, 'predicted', k)
            row.append(v)
        for k in edge_keys:
            v = _get_mean(summary, 'edge_metrics', k)
            row.append(v)
        raw_vals.append(row)

    # Smart UV row (from first summary, assumed consistent)
    smart_row = []
    for k in metric_keys:
        vs = [_get_mean(s, 'smart_uv', k) for s in summaries]
        valid = [v for v in vs if not np.isnan(v)]
        smart_row.append(float(np.mean(valid)) if valid else float('nan'))
    smart_row += [float('nan')] * len(edge_keys)
    raw_vals.append(smart_row)

    # GT row
    gt_row = []
    for k in metric_keys:
        vs = [_get_mean(s, 'ground_truth', k) for s in summaries]
        valid = [v for v in vs if not np.isnan(v)]
        gt_row.append(float(np.mean(valid)) if valid else float('nan'))
    gt_row += [float('nan')] * len(edge_keys)
    raw_vals.append(gt_row)

    raw_arr = np.array(raw_vals, dtype=float)

    n_model_rows = len(labels)
    best_per_col = []
    for col in range(n_cols):
        col_vals = raw_arr[:n_model_rows + 1, col]  # models + smart_uv
        if col < len(metric_keys):
            if not np.all(np.isnan(col_vals)):
                best_row = int(np.nanargmin(col_vals))
                best_per_col.append(best_row)
            else:
                best_per_col.append(-1)
        else:
            if not np.all(np.isnan(col_vals)):
                best_row = int(np.nanargmax(col_vals))
                best_per_col.append(best_row)
            else:
                best_per_col.append(-1)

    for row_idx, row in enumerate(raw_arr):
        cell_row = []
        for col_idx, v in enumerate(row):
            cell_row.append('N/A' if np.isnan(v) else f'{v:.4f}')
        cell_data.append(cell_row)

    fig_h = max(4, 0.45 * n_rows + 1.5)
    fig_w = max(12, 1.2 * n_cols + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    col_labels = display_metrics
    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='right',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for col in range(n_cols):
        tbl[0, col].set_facecolor('#1565C0')
        tbl[0, col].set_text_props(color='white', fontweight='bold')

    for col_idx, best_row in enumerate(best_per_col):
        if best_row >= 0:
            tbl[best_row + 1, col_idx].set_facecolor('#BBDEFB')

    gt_row_idx = n_rows
    for col_idx in range(n_cols):
        tbl[gt_row_idx, col_idx].set_facecolor('#E8F5E9')

    plt.title('Full Method Comparison — UV Quality & Edge Metrics',
              fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI, bbox_inches='tight')
    plt.close()
    print(f'[compare] saved: {output_path}')


def plot_method_comparison_bars(summaries: list[dict], labels: list[str], output_path: str) -> None:
    import matplotlib.pyplot as plt

    metrics = ['area_distortion_avg', 'angle_distortion_avg', 'symmetric_dirichlet_avg']
    display = ['Area Distortion', 'Angle Distortion', 'Sym. Dirichlet']

    method_labels = labels + ['Smart UV', 'Ground Truth']
    model_colors = _COLORS_MODEL[:len(labels)] + [_C_SMART, _C_GT]

    values = []
    for m_idx, (label, color) in enumerate(zip(method_labels, model_colors)):
        row = []
        if m_idx < len(summaries):
            for k in metrics:
                row.append(_get_mean(summaries[m_idx], 'predicted', k))
        elif label == 'Smart UV':
            for k in metrics:
                vs = [_get_mean(s, 'smart_uv', k) for s in summaries]
                valid = [v for v in vs if not np.isnan(v)]
                row.append(float(np.mean(valid)) if valid else 0.0)
        else:  # GT
            for k in metrics:
                vs = [_get_mean(s, 'ground_truth', k) for s in summaries]
                valid = [v for v in vs if not np.isnan(v)]
                row.append(float(np.mean(valid)) if valid else 0.0)
        values.append(row)

    x = np.arange(len(metrics))
    n_methods = len(method_labels)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, color, row) in enumerate(zip(method_labels, model_colors, values)):
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, row, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylabel('Distortion (lower is better)')
    ax.set_title('UV Distortion — All Methods', fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI)
    plt.close()
    print(f'[compare] saved: {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Cross-model UV evaluation comparison.')
    parser.add_argument('result_dirs', nargs='+', help='evaluation/results/... directories')
    parser.add_argument('--output-dir', required=True, help='Where to save comparison plots')
    args = parser.parse_args()

    summaries, labels, dirs = [], [], []
    for d in args.result_dirs:
        s = _load_summary(d)
        if s is not None:
            summaries.append(s)
            labels.append(_model_label(s, d))
            dirs.append(d)

    if not summaries:
        print('[compare] no valid result directories found.')
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_full_comparison_table(summaries, labels, str(out_dir / 'full_comparison_table.png'))
    plot_method_comparison_bars(summaries, labels, str(out_dir / 'method_comparison_bars.png'))

    merged = {
        'models': [
            {'label': lbl, 'dir': d, 'summary': s}
            for lbl, d, s in zip(labels, dirs, summaries)
        ]
    }
    with open(out_dir / 'comparison_summary.json', 'w') as f:
        json.dump(merged, f, indent=2, default=str)

    print(f'[compare] done — {len(summaries)} model(s) compared.')


if __name__ == '__main__':
    main()
