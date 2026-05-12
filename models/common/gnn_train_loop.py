from __future__ import annotations

import torch
from torch_geometric.data import Data

from models.utils.losses import focal_bce_with_logits
from models.utils.metrics import RECALL_TPR_LABEL, edge_f1


def metric_line(label: str, loss: float | None, metrics: dict) -> str:
    loss_part = f"loss {loss:.4f}  " if loss is not None else ''
    return (
        f"{label} | {loss_part}f1 {metrics['f1']:.4f}  "
        f"prec {metrics['precision']:.4f}  {RECALL_TPR_LABEL} {metrics['recall']:.4f}  "
        f"fpr {metrics['fpr']:.4f}  acc {metrics['accuracy']:.4f}"
    )


def confusion_counts(metrics: dict) -> dict:
    return {key: int(metrics[key]) for key in ('tp', 'fp', 'fn', 'tn')}


def run_epoch(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
    pos_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    focal_gamma: float = 2.0,
) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_logits, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for data in graphs:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)

            logits = model(x, edge_index)
            loss = focal_bce_with_logits(logits, y, pos_weight, focal_gamma)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.cpu())

            del x, edge_index, y, logits, loss
            torch.cuda.empty_cache()

    mean_loss = total_loss / len(graphs)
    metrics = edge_f1(torch.cat(all_logits), torch.cat(all_labels))
    return mean_loss, metrics


def collect_logits_labels(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, labels = [], []
    with torch.no_grad():
        for data in graphs:
            logits.append(model(data.x.to(device), data.edge_index.to(device)).cpu())
            labels.append(data.y.cpu())
    return torch.cat(logits), torch.cat(labels)


def print_threshold_sweep(title: str, rows: list[dict], best_t: float, marker_label: str) -> None:
    print(title)
    print(f"  {'t':>5s}  {'P':>7s}  {RECALL_TPR_LABEL:>8s}  {'F1':>7s}  {'FPR':>7s}")
    for row in rows:
        marker = marker_label if row['threshold'] == best_t else ''
        print(
            f"  {row['threshold']:>5.2f}  {row['precision']:>7.4f}  "
            f"{row['recall']:>8.4f}  {row['f1']:>7.4f}  {row['fpr']:>7.4f}{marker}"
        )
