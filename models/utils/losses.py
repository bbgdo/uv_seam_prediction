import torch
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = labels * probs + (1 - labels) * (1 - probs)
    focal_weight = (1 - p_t) ** gamma
    class_weight = labels * pos_weight + (1 - labels)
    return (focal_weight * class_weight * bce).mean()
