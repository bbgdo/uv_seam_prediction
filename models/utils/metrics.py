import torch


@torch.no_grad()
def edge_f1(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    preds = (torch.sigmoid(logits) >= threshold).long()
    gt = labels.long()

    tp = (preds & gt).sum().item()
    fp = (preds & ~gt.bool()).sum().item()
    fn = (~preds.bool() & gt.bool()).sum().item()
    tn = (~preds.bool() & ~gt.bool()).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(gt), 1)

    return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}
