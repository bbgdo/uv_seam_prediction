import torch


DEFAULT_THRESHOLD_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
RECALL_TPR_LABEL = 'rec(tpr)'


@torch.no_grad()
def threshold_sweep(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_VALUES,
) -> dict:
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
    tpr = recall

    return {
        'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy,
        'fpr': fpr, 'tpr': tpr, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }
