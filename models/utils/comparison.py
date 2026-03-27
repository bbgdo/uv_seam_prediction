import json
import sys
from pathlib import Path


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    summary_path = run_dir / 'summary.json'
    metrics_path = run_dir / 'metrics.json'

    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    metrics = []
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return summary, metrics


def plot_comparison_f1(runs: dict[str, tuple[dict, list[dict]]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from models.utils.experiment_log import _apply_style, FIGSIZE_LINE, DPI, FONT_LABEL, FONT_TICK
    _apply_style(plt)

    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']

    for i, (name, (summary, metrics)) in enumerate(runs.items()):
        if not metrics:
            continue
        epochs = [m['epoch'] for m in metrics]
        val_f1 = [m.get('val_f1', 0) for m in metrics]
        best_f1 = summary.get('best_val_f1', max(val_f1, default=0))
        color = colors[i % len(colors)]
        ax.plot(epochs, val_f1, color=color, label=f'{name} (best={best_f1:.4f})', linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=FONT_LABEL)
    ax.set_ylabel('Val F1', fontsize=FONT_LABEL)
    ax.set_title('Model Comparison — Validation F1', fontsize=FONT_LABEL + 2)
    ax.legend(fontsize=FONT_TICK)
    ax.tick_params(labelsize=FONT_TICK)
    fig.tight_layout()
    fig.savefig(output_dir / 'comparison_f1.png', dpi=DPI)
    plt.close(fig)
    print(f"  saved comparison_f1.png")


def plot_comparison_table(runs: dict[str, tuple[dict, list[dict]]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from models.utils.experiment_log import DPI, FONT_TICK

    columns = ['Model', 'F1', 'Precision', 'Recall', 'Accuracy', 'Best Epoch']
    rows = []
    for name, (summary, _) in runs.items():
        rows.append([
            name,
            f"{summary.get('test_f1', 0):.4f}",
            f"{summary.get('test_precision', 0):.4f}",
            f"{summary.get('test_recall', 0):.4f}",
            f"{summary.get('test_accuracy', 0):.4f}",
            str(summary.get('best_epoch', '-')),
        ])

    fig, ax = plt.subplots(figsize=(10, 1 + 0.5 * len(rows)))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_TICK)
    table.scale(1, 1.5)

    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#e0e0e0')
        table[(0, j)].set_text_props(weight='bold')

    ax.set_title('Test Results Comparison', fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(output_dir / 'comparison_table.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved comparison_table.png")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python models/utils/comparison.py <run_dir1> <run_dir2> [run_dir3 ...]")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    run_dirs = [Path(p) for p in sys.argv[1:]]
    for d in run_dirs:
        if not d.exists():
            print(f"[error] not found: {d}")
            sys.exit(1)

    runs = {}
    for d in run_dirs:
        name = d.name
        summary, metrics = load_run(d)
        runs[name] = (summary, metrics)
        best_f1 = summary.get('best_val_f1', 'N/A')
        test_f1 = summary.get('test_f1', 'N/A')
        print(f"  loaded {name}: {len(metrics)} epochs, best val F1={best_f1}, test F1={test_f1}")

    output_dir = run_dirs[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_comparison_f1(runs, output_dir)
    plot_comparison_table(runs, output_dir)
    print(f"\ncomparison plots saved -> {output_dir}")
