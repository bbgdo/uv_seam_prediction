import torch
import torch.nn.functional as F


def connectivity_penalty(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Penalty = mean of prob_self * (1 - mean_prob_neighbors).

    Discourages isolated high-probability seam predictions whose
    neighbors have low probability.
    """
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
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    conn = connectivity_penalty(logits, edge_index)
    return bce + lambda_conn * conn
