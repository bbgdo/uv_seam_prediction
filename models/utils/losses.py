import torch
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss — down-weights easy examples, better for extreme class imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = labels * probs + (1 - labels) * (1 - probs)
    focal_weight = (1 - p_t) ** gamma
    class_weight = labels * pos_weight + (1 - labels)
    return (focal_weight * class_weight * bce).mean()


def connectivity_penalty(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Penalises isolated seam predictions whose neighbors have low probability."""
    probs = torch.sigmoid(logits)
    src, dst = edge_index
    neighbor_sum = torch.zeros_like(probs)
    neighbor_count = torch.zeros_like(probs)
    neighbor_sum.scatter_add_(0, src, probs[dst])
    neighbor_count.scatter_add_(0, src, torch.ones_like(probs[dst]))
    neighbor_mean = neighbor_sum / neighbor_count.clamp(min=1)
    isolation = probs * (1.0 - neighbor_mean)
    return isolation.mean()


def seam_loss_with_connectivity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    pos_weight: torch.Tensor,
    lambda_conn: float = 0.1,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    loss = focal_bce_with_logits(logits, labels, pos_weight, focal_gamma)
    conn = connectivity_penalty(logits, edge_index)
    return loss + lambda_conn * conn
