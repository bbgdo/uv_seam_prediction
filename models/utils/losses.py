import torch
import torch.nn.functional as F


def connectivity_penalty(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Penalize isolated seam predictions on the dual graph.

    A predicted seam node whose dual-graph neighbors have low seam probability
    is unlikely to form a useful UV boundary — penalize it.
    Penalty = mean over nodes of: prob_self * (1 - mean_prob_neighbors).
    """
    probs = torch.sigmoid(logits)
    src, dst = edge_index

    neighbor_sum = torch.zeros_like(probs)
    neighbor_count = torch.zeros_like(probs)
    neighbor_sum.scatter_add_(0, src, probs[dst])
    neighbor_count.scatter_add_(0, src, torch.ones_like(probs[dst]))

    neighbor_mean = neighbor_sum / neighbor_count.clamp(min=1)

    # high own prob + low neighbor mean = isolated prediction = bad
    isolation = probs * (1.0 - neighbor_mean)
    return isolation.mean()


def seam_loss_with_connectivity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    pos_weight: torch.Tensor,
    lambda_conn: float = 0.1,
) -> torch.Tensor:
    """BCE loss + weighted connectivity penalty."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    conn = connectivity_penalty(logits, edge_index)
    return bce + lambda_conn * conn
