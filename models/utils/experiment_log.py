"""
Experiment logging: stores per-epoch metrics as JSON, generates plots as PNG.

Usage:
    logger = ExperimentLogger('runs/gatv2_exp01', config={...})
    for epoch in range(100):
        logger.log_epoch(epoch, train_loss=0.5, val_loss=0.4, val_f1=0.8, ...)
    logger.finalize(test_metrics={...}, best_epoch=42)
    logger.save()
    logger.plot()
"""

import json
import time
from pathlib import Path

import torch
from torch_geometric.data import Data


# plot style constants
COLOR_TRAIN = '#2196F3'
COLOR_VAL = '#FF5722'
COLOR_TEST = '#4CAF50'
FIGSIZE_LINE = (10, 6)
FIGSIZE_BAR = (8, 5)
DPI = 150
FONT_LABEL = 12
FONT_TICK = 10


class ExperimentLogger:
    def __init__(self, run_dir: str | Path, config: dict | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.metrics: list[dict] = []
        self.summary: dict = {}
        self._start_time = time.time()

    def log_epoch(self, epoch: int, **kwargs) -> None:
        """Log metrics for one epoch. Pass any key=value pairs."""
        entry = {'epoch': epoch, **kwargs}
        self.metrics.append(entry)

    def log_class_balance(
        self,
        train: list[Data],
        val: list[Data],
        test: list[Data],
    ) -> None:
        """Log and plot class balance across splits."""
        def _count(graphs):
            seam = sum(d.y.sum().item() for d in graphs)
            total = sum(len(d.y) for d in graphs)
            return int(seam), int(total - seam)

        self._class_balance = {
            'train': _count(train),
            'val': _count(val),
            'test': _count(test),
        }

    def finalize(self, test_metrics: dict, best_epoch: int) -> None:
        """Record final summary after training."""
        total_time = time.time() - self._start_time
        best_entry = None
        for entry in self.metrics:
            if entry.get('epoch') == best_epoch:
                best_entry = entry
                break

        self.summary = {
            'model': self.config.get('model', 'unknown'),
            'best_epoch': best_epoch,
            'best_val_f1': best_entry.get('val_f1', 0) if best_entry else 0,
            'best_val_precision': best_entry.get('val_precision', 0) if best_entry else 0,
            'best_val_recall': best_entry.get('val_recall', 0) if best_entry else 0,
            'test_f1': test_metrics.get('f1', 0),
            'test_precision': test_metrics.get('precision', 0),
            'test_recall': test_metrics.get('recall', 0),
            'test_accuracy': test_metrics.get('accuracy', 0),
            'total_epochs': len(self.metrics),
            'total_time_s': round(total_time, 1),
            'dataset_stats': {
                'train_graphs': self.config.get('train_graphs', 0),
                'val_graphs': self.config.get('val_graphs', 0),
                'test_graphs': self.config.get('test_graphs', 0),
                'pos_weight': self.config.get('pos_weight', 0),
            },
        }

    def save(self) -> None:
        """Write config, metrics, and summary to JSON files."""
        _write_json(self.run_dir / 'config.json', self.config)
        _write_json(self.run_dir / 'metrics.json', self.metrics)
        if self.summary:
            _write_json(self.run_dir / 'summary.json', self.summary)
        print(f"logs saved -> {self.run_dir}")

    def plot(self) -> None:
        """Generate all training plots as PNG."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warning] matplotlib not installed, skipping plots")
            return

        _apply_style(plt)

        epochs = [m['epoch'] for m in self.metrics]
        best_epoch = self.summary.get('best_epoch')

        self._plot_loss(plt, epochs, best_epoch)
        self._plot_f1(plt, epochs, best_epoch)
        self._plot_precision_recall(plt, epochs, best_epoch)
        self._plot_lr(plt, epochs)
        if hasattr(self, '_class_balance'):
            self._plot_class_balance(plt)

        print(f"plots saved -> {self.run_dir}")

    def _plot_loss(self, plt, epochs, best_epoch) -> None:
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
        train_loss = [m.get('train_loss', 0) for m in self.metrics]
        val_loss = [m.get('val_loss', 0) for m in self.metrics]

        ax.plot(epochs, train_loss, color=COLOR_TRAIN, label='Train loss', linewidth=1.5)
        ax.plot(epochs, val_loss, color=COLOR_VAL, label='Val loss', linewidth=1.5)

        # use log scale if range spans >2 orders of magnitude
        all_vals = [v for v in train_loss + val_loss if v > 0]
        if all_vals and max(all_vals) / (min(all_vals) + 1e-12) > 100:
            ax.set_yscale('log')

        ax.set_xlabel('Epoch', fontsize=FONT_LABEL)
        ax.set_ylabel('Loss', fontsize=FONT_LABEL)
        ax.set_title('Training and Validation Loss', fontsize=FONT_LABEL + 2)
        ax.legend(fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK)
        fig.tight_layout()
        fig.savefig(self.run_dir / 'loss_curves.png', dpi=DPI)
        plt.close(fig)

    def _plot_f1(self, plt, epochs, best_epoch) -> None:
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
        train_f1 = [m.get('train_f1', 0) for m in self.metrics]
        val_f1 = [m.get('val_f1', 0) for m in self.metrics]

        ax.plot(epochs, train_f1, color=COLOR_TRAIN, label='Train F1', linewidth=1.5)
        ax.plot(epochs, val_f1, color=COLOR_VAL, label='Val F1', linewidth=1.5)

        if best_epoch is not None:
            best_f1 = max((m.get('val_f1', 0) for m in self.metrics), default=0)
            ax.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7)
            ax.annotate(
                f'Best: {best_f1:.4f}',
                xy=(best_epoch, best_f1),
                xytext=(best_epoch + 1, best_f1 - 0.05),
                fontsize=FONT_TICK,
                arrowprops=dict(arrowstyle='->', color='gray'),
            )

        ax.set_xlabel('Epoch', fontsize=FONT_LABEL)
        ax.set_ylabel('F1 Score', fontsize=FONT_LABEL)
        ax.set_title('Training and Validation F1', fontsize=FONT_LABEL + 2)
        ax.legend(fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK)
        fig.tight_layout()
        fig.savefig(self.run_dir / 'f1_curves.png', dpi=DPI)
        plt.close(fig)

    def _plot_precision_recall(self, plt, epochs, best_epoch) -> None:
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
        val_prec = [m.get('val_precision', 0) for m in self.metrics]
        val_rec = [m.get('val_recall', 0) for m in self.metrics]

        ax.plot(epochs, val_prec, color=COLOR_VAL, label='Val Precision', linewidth=1.5)
        ax.plot(epochs, val_rec, color=COLOR_TEST, label='Val Recall', linewidth=1.5)

        ax.set_xlabel('Epoch', fontsize=FONT_LABEL)
        ax.set_ylabel('Score', fontsize=FONT_LABEL)
        ax.set_title('Validation Precision and Recall', fontsize=FONT_LABEL + 2)
        ax.legend(fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK)
        fig.tight_layout()
        fig.savefig(self.run_dir / 'precision_recall_curves.png', dpi=DPI)
        plt.close(fig)

    def _plot_lr(self, plt, epochs) -> None:
        lr_vals = [m.get('lr') for m in self.metrics]
        if not any(v is not None for v in lr_vals):
            return

        fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
        ax.plot(epochs, lr_vals, color=COLOR_TRAIN, linewidth=1.5)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=FONT_LABEL)
        ax.set_ylabel('Learning Rate', fontsize=FONT_LABEL)
        ax.set_title('Learning Rate Schedule', fontsize=FONT_LABEL + 2)
        ax.tick_params(labelsize=FONT_TICK)
        fig.tight_layout()
        fig.savefig(self.run_dir / 'lr_schedule.png', dpi=DPI)
        plt.close(fig)

    def _plot_class_balance(self, plt) -> None:
        fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
        splits = ['Train', 'Val', 'Test']
        keys = ['train', 'val', 'test']
        seam_counts = [self._class_balance[k][0] for k in keys]
        nonseam_counts = [self._class_balance[k][1] for k in keys]

        x = range(len(splits))
        width = 0.35
        ax.bar([i - width / 2 for i in x], seam_counts, width, label='Seam', color=COLOR_VAL)
        ax.bar([i + width / 2 for i in x], nonseam_counts, width, label='Non-seam', color=COLOR_TRAIN)

        ax.set_xticks(list(x))
        ax.set_xticklabels(splits, fontsize=FONT_TICK)
        ax.set_ylabel('Edge count', fontsize=FONT_LABEL)
        ax.set_title('Class Balance by Split', fontsize=FONT_LABEL + 2)
        ax.legend(fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK)
        fig.tight_layout()
        fig.savefig(self.run_dir / 'class_balance.png', dpi=DPI)
        plt.close(fig)


def _write_json(path: Path, data) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def _apply_style(plt) -> None:
    """Apply a clean plot style, falling back gracefully."""
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        try:
            plt.style.use(style)
            return
        except OSError:
            continue
