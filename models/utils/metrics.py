import torch

from models.common.config import DEFAULT_THRESHOLD_VALUES


RECALL_TPR_LABEL = 'rec(tpr)'


def metric_display_label(metric: str) -> str:
    if metric == 'recall':
        return RECALL_TPR_LABEL
    return metric


@torch.no_grad()
def threshold_sweep(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_VALUES,
) -> dict:
    """Evaluate F1 across thresholds, return best threshold and full results."""
    results = []
    for t in thresholds:
        m = edge_f1(logits, labels, threshold=t)
        m['threshold'] = t
        results.append(m)
    best = max(results, key=lambda r: r['f1'])
    return {'best': best, 'all': results}


@torch.no_grad()
def edge_f1(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    return binary_metrics_from_probs(torch.sigmoid(logits), labels, threshold)


@torch.no_grad()
def binary_metrics_from_probs(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).long()
    gt = labels.long()

    tp = (preds & gt).sum().item()
    fp = (preds & ~gt.bool()).sum().item()
    fn = (~preds.bool() & gt.bool()).sum().item()
    tn = (~preds.bool() & ~gt.bool()).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(gt), 1)
    fpr = fp / max(fp + tn, 1)
    tpr = recall  # same as recall

    return {
        'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy,
        'fpr': fpr, 'tpr': tpr, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }
