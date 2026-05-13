from __future__ import annotations

from typing import Any

import torch

from tools.utils.reeval_common import SavedRun, metric_delta, split_identity
from tools.utils.reeval_thresholds import exact_validation_threshold, metrics_at_threshold, threshold_table


def build_reevaluation_payload(
    *,
    target: SavedRun,
    model_name: str,
    display_name: str,
    feature_selection: dict[str, Any],
    split_info: dict[str, Any],
    val_graphs: int,
    test_graphs: int,
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    test_logits: torch.Tensor,
    test_labels: torch.Tensor,
    report_grid: list[float],
    threshold_decimals: int | None,
    old_validation_best: dict[str, Any] | None,
) -> dict[str, Any]:
    val_probs = torch.sigmoid(val_logits)
    test_probs = torch.sigmoid(test_logits)

    exact = exact_validation_threshold(val_probs, val_labels, threshold_decimals)
    exact_threshold = float(exact['threshold'])
    old_threshold = target.summary.get('best_validation_threshold')
    if old_threshold is None and old_validation_best:
        old_threshold = old_validation_best.get('threshold')

    val_metrics = {
        'threshold_0_5': metrics_at_threshold(val_probs, val_labels, 0.5),
        'exact_val_best': exact['metrics'],
    }
    test_metrics = {
        'threshold_0_5': metrics_at_threshold(test_probs, test_labels, 0.5),
        'exact_val_best': metrics_at_threshold(test_probs, test_labels, exact_threshold),
    }
    if old_threshold is not None:
        old_threshold = float(old_threshold)
        val_metrics['old_val_best'] = metrics_at_threshold(val_probs, val_labels, old_threshold)
        test_metrics['old_val_best'] = metrics_at_threshold(test_probs, test_labels, old_threshold)
    else:
        val_metrics['old_val_best'] = None
        test_metrics['old_val_best'] = None

    old_stored = {
        'validation_val_best': old_validation_best,
        'test_val_best': target.summary.get('test_metrics_best_validation_threshold'),
        'test_0_5': target.summary.get('test_metrics_threshold_0_5'),
    }

    payload = {
        'status': 'completed',
        'run_identity': {
            'experiment': target.experiment,
            'seed': target.seed,
            'run_dir': str(target.run_dir),
        },
        'checkpoint_path': str(target.checkpoint_path),
        'split_path': str(target.split_path),
        'dataset_path': str(target.dataset_path),
        'model_family': {
            'model_name': model_name,
            'display_name': display_name,
        },
        'feature_selection': feature_selection,
        'split': {
            'seed': split_info.get('seed'),
            'resolution_tag': split_info.get('resolution_tag'),
            'val_graphs': val_graphs,
            'test_graphs': test_graphs,
        },
        'threshold_search': {
            'method': 'validation_f1_threshold_search',
            'threshold_decimals': threshold_decimals,
            'candidate_source': exact['candidate_source'],
            'candidate_count': exact['candidate_count'],
            'tie_breaking': exact['tie_breaking'],
            'dense_report_grid': report_grid,
        },
        'old_threshold': old_threshold,
        'exact_validation_optimal_threshold': exact_threshold,
        'metrics': {
            'validation': val_metrics,
            'test': test_metrics,
        },
        'dense_grid': {
            'validation': threshold_table(val_probs, val_labels, report_grid),
            'test': threshold_table(test_probs, test_labels, report_grid),
        },
        'old_stored_metrics': old_stored,
        'comparison': {
            'delta_vs_old_stored_val_best': {
                'validation': metric_delta(val_metrics['exact_val_best'], old_stored['validation_val_best']),
                'test': metric_delta(test_metrics['exact_val_best'], old_stored['test_val_best']),
            },
            'delta_vs_0_5': {
                'validation': metric_delta(val_metrics['exact_val_best'], val_metrics['threshold_0_5']),
                'test': metric_delta(test_metrics['exact_val_best'], test_metrics['threshold_0_5']),
            },
        },
    }
    payload['split_identity'] = split_identity(payload)
    return payload
