from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from models.utils.metrics import binary_metrics_from_probs
from tools.utils.reeval_common import DEFAULT_REPORT_GRID, TIE_BREAKING


def build_report_grid(spec: str | None = None) -> list[float]:
    if spec is None or not spec.strip():
        values = list(DEFAULT_REPORT_GRID)
    elif ':' in spec and ',' not in spec:
        parts = [float(part) for part in spec.split(':')]
        if len(parts) != 3:
            raise ValueError('--report-grid range must be start:stop:step')
        start, stop, step = parts
        if step <= 0:
            raise ValueError('--report-grid step must be positive')
        values = []
        current = start
        while current <= stop + (step / 2):
            values.append(round(current, 12))
            current += step
    else:
        values = [float(part.strip()) for part in spec.split(',') if part.strip()]

    cleaned = []
    for value in values:
        if value < 0 or value > 1:
            raise ValueError(f'threshold must be between 0 and 1, got {value}')
        if math.isclose(value, 1.0):
            continue
        cleaned.append(float(value))

    grid = sorted(set(cleaned))
    if not grid:
        raise ValueError('report grid is empty after excluding threshold 1.0')
    return grid


def metrics_at_threshold(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, Any]:
    metrics = binary_metrics_from_probs(probs, labels, threshold=float(threshold))
    metrics['threshold'] = float(threshold)
    return metrics


def threshold_table(probs: torch.Tensor, labels: torch.Tensor, thresholds: list[float]) -> list[dict[str, Any]]:
    return [metrics_at_threshold(probs, labels, threshold) for threshold in thresholds]


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def best_threshold_index(
    *,
    f1: np.ndarray,
    fpr: np.ndarray,
    precision: np.ndarray,
    thresholds: np.ndarray,
) -> int:
    order = np.lexsort((thresholds, precision, -fpr, f1))
    return int(order[-1])


def compute_threshold_metrics_fast(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    prob_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    label_arr = np.asarray(labels).reshape(-1)
    if prob_arr.size == 0:
        raise ValueError('cannot search threshold on an empty validation set')
    if prob_arr.size != label_arr.size:
        raise ValueError('probabilities and labels must have the same number of elements')

    labels_bool = label_arr.astype(bool)
    order = np.argsort(-prob_arr, kind='mergesort')
    sorted_probs = prob_arr[order]
    sorted_labels = labels_bool[order]

    group_ends = np.flatnonzero(np.r_[sorted_probs[1:] != sorted_probs[:-1], True])
    thresholds = sorted_probs[group_ends]
    valid = thresholds < 1.0
    thresholds = thresholds[valid]
    group_ends = group_ends[valid]

    cumulative_pos = np.cumsum(sorted_labels, dtype=np.int64)
    tp = cumulative_pos[group_ends] if group_ends.size else np.array([], dtype=np.int64)
    predicted_positive = group_ends.astype(np.int64) + 1
    fp = predicted_positive - tp

    max_score = float(np.max(prob_arr))
    above_max = float(np.nextafter(np.float64(max_score), np.float64(1.0)))
    if max_score < 1.0 and above_max < 1.0:
        thresholds = np.r_[above_max, thresholds]
        tp = np.r_[np.int64(0), tp]
        fp = np.r_[np.int64(0), fp]
    elif thresholds.size == 0:
        near_one_threshold = float(np.nextafter(np.float64(1.0), np.float64(0.0)))
        positive_at_threshold = prob_arr >= near_one_threshold
        thresholds = np.array([near_one_threshold], dtype=np.float64)
        tp = np.array([np.count_nonzero(positive_at_threshold & labels_bool)], dtype=np.int64)
        fp = np.array([np.count_nonzero(positive_at_threshold & ~labels_bool)], dtype=np.int64)

    total = int(prob_arr.size)
    total_pos = int(np.count_nonzero(labels_bool))
    total_neg = total - total_pos
    fn = total_pos - tp
    tn = total_neg - fp

    precision = safe_divide(tp.astype(np.float64), (tp + fp).astype(np.float64))
    recall = safe_divide(tp.astype(np.float64), (tp + fn).astype(np.float64))
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-8)
    fpr = safe_divide(fp.astype(np.float64), (fp + tn).astype(np.float64))
    accuracy = (tp + tn).astype(np.float64) / max(total, 1)

    best_index = best_threshold_index(
        f1=f1,
        fpr=fpr,
        precision=precision,
        thresholds=thresholds,
    )
    best_metrics = {
        'f1': float(f1[best_index]),
        'precision': float(precision[best_index]),
        'recall': float(recall[best_index]),
        'accuracy': float(accuracy[best_index]),
        'fpr': float(fpr[best_index]),
        'tpr': float(recall[best_index]),
        'tp': int(tp[best_index]),
        'fp': int(fp[best_index]),
        'fn': int(fn[best_index]),
        'tn': int(tn[best_index]),
        'threshold': float(thresholds[best_index]),
    }

    return {
        'threshold': float(thresholds[best_index]),
        'metrics': best_metrics,
        'candidate_count': int(thresholds.size),
        'candidate_source': (
            'unique validation score breakpoints below 1.0 using cumulative counts; '
            'includes nextafter(max_score, 1.0) no-positive breakpoint when below 1.0'
        ),
        'tie_breaking': list(TIE_BREAKING),
    }


def exact_validation_threshold(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    flat_probs = probs.detach().flatten().cpu().numpy()
    flat_labels = labels.detach().flatten().cpu().numpy()
    return compute_threshold_metrics_fast(flat_probs, flat_labels)
