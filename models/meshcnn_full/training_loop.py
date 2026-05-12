from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import Any

import torch

from models.meshcnn_full.mesh import MeshCNNSample
from models.meshcnn_full.model import MeshCNNSegmenter
from models.utils.losses import focal_bce_with_logits
from models.utils.metrics import edge_f1


SAFE_DIV_EPS = 1e-8


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den + SAFE_DIV_EPS)


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def collect_pool_debug(model):
    out = []
    for module in model.modules():
        getter = getattr(module, 'get_last_debug', None)
        if getter is None:
            continue
        debug = getter()
        if debug:
            out.append(debug)
    return out


def coerce_id_list(value) -> list[int] | None:
    if value is None:
        return None
    if callable(value):
        try:
            value = value()
        except TypeError:
            return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().view(-1).tolist()
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def restored_orig_edge_ids(model) -> list[int] | None:
    for name in (
        'get_restored_orig_edge_ids',
        'get_last_restored_orig_edge_ids',
        'restored_orig_edge_ids',
        'last_restored_orig_edge_ids',
    ):
        value = getattr(model, name, None)
        ids = coerce_id_list(value)
        if ids is not None:
            return ids
    return None


def append_seam_enrichment(
    epoch_debug,
    layer_idx: int,
    labels: torch.Tensor,
    survivor_orig_edge_ids,
) -> None:
    survivor_ids = coerce_id_list(survivor_orig_edge_ids)
    if survivor_ids is None:
        return

    labels_cpu = labels.detach().cpu().view(-1)
    label_count = int(labels_cpu.numel())
    if label_count == 0:
        return

    valid_ids = sorted({idx for idx in survivor_ids if 0 <= idx < label_count})
    survivor_mask = torch.zeros(label_count, dtype=torch.bool)
    if valid_ids:
        survivor_mask[torch.as_tensor(valid_ids, dtype=torch.long)] = True

    seam_mask = labels_cpu > 0.5
    nonseam_mask = ~seam_mask
    total_gt_seams = int(seam_mask.sum().item())
    total_nonseams = int(nonseam_mask.sum().item())
    surviving_gt_seams = int((survivor_mask & seam_mask).sum().item())
    surviving_nonseams = int((survivor_mask & nonseam_mask).sum().item())

    gt_retention = safe_div(surviving_gt_seams, total_gt_seams)
    nonseam_retention = safe_div(surviving_nonseams, total_nonseams)
    seam_enrichment = gt_retention / max(nonseam_retention, 1e-8)

    epoch_debug[f'pool/layer{layer_idx}_gt_seam_retention'].append(gt_retention)
    epoch_debug[f'pool/layer{layer_idx}_nonseam_retention'].append(nonseam_retention)
    epoch_debug[f'pool/layer{layer_idx}_seam_enrichment'].append(float(seam_enrichment))


def append_debug_metrics(
    epoch_debug,
    pool_debug_list,
    model,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    pool_metric_keys = (
        'input_edges',
        'output_edges',
        'successful_collapses',
        'rejected_collapses',
        'reject_boundary',
        'reject_nonmanifold',
        'reject_degenerate',
        'collapsed_norm_mean',
        'retained_norm_mean',
    )
    for layer_idx, debug in enumerate(pool_debug_list):
        append_seam_enrichment(
            epoch_debug,
            layer_idx,
            labels,
            debug.get('survivor_orig_edge_ids'),
        )
        for key in pool_metric_keys:
            epoch_debug[f'pool/layer{layer_idx}_{key}'].append(debug.get(key))

    expected_count = int(labels.numel())
    restored_ids = restored_orig_edge_ids(model)
    if restored_ids is not None:
        restored_count = len(restored_ids)
        unique_count = len(set(restored_ids))
        missing_edge_ids = expected_count - unique_count
        duplicate_edge_ids = restored_count - unique_count
        final_edge_count_mismatch = abs(restored_count - expected_count)
    else:
        missing_edge_ids = None
        duplicate_edge_ids = None
        final_edge_count_mismatch = abs(int(logits.numel()) - expected_count)

    epoch_debug['unpool/missing_edge_ids'].append(missing_edge_ids)
    epoch_debug['unpool/duplicate_edge_ids'].append(duplicate_edge_ids)
    epoch_debug['unpool/final_edge_count_mismatch'].append(final_edge_count_mismatch)


def finalize_debug(epoch_debug) -> dict[str, float]:
    out: dict[str, float] = {}
    for tag, values in epoch_debug.items():
        value = mean_or_none(values)
        if value is not None:
            out[tag] = float(value)
    return out


def log_debug_scalars(writer, debug_metrics: dict[str, float], epoch: int, prefix: str = '') -> None:
    if writer is None:
        return
    for tag, value in debug_metrics.items():
        writer.add_scalar(f'{prefix}/{tag}' if prefix else tag, value, epoch)


def format_debug_summary(debug_metrics: dict[str, float], metrics: dict[str, Any]) -> str:
    parts = []
    pred_pos_rate = metrics.get('pred_pos_rate')
    if pred_pos_rate is not None:
        parts.append(f'pred_pos_rate {pred_pos_rate:.4f}')
    layer0_enrichment = debug_metrics.get('pool/layer0_seam_enrichment')
    if layer0_enrichment is not None:
        parts.append(f'layer0_seam_enrichment {layer0_enrichment:.4f}')
    mismatch = debug_metrics.get('unpool/final_edge_count_mismatch')
    if mismatch is not None:
        parts.append(f'unpool_mismatch {mismatch:.2f}')
    return ' | '.join(parts)


def run_epoch(
    model: MeshCNNSegmenter,
    samples: list[MeshCNNSample],
    device: torch.device,
    pos_weight: torch.Tensor,
    focal_gamma: float,
    optimizer: torch.optim.Optimizer | None = None,
    grad_accum_steps: int = 1,
    progress_prefix: str | None = None,
) -> tuple[float, dict[str, Any], dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    epoch_debug = defaultdict(list)
    is_tty = sys.stdout.isatty()

    if training:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for idx, sample in enumerate(samples, start=1):
            step_t0 = time.time()
            labels = sample.edge_labels.to(device)
            logits = model(sample)
            pool_debug_list = collect_pool_debug(model)
            append_debug_metrics(epoch_debug, pool_debug_list, model, logits, labels)
            loss = focal_bce_with_logits(logits, labels, pos_weight, gamma=focal_gamma)

            if training:
                (loss / max(grad_accum_steps, 1)).backward()
                if idx % grad_accum_steps == 0 or idx == len(samples):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            step_time = time.time() - step_t0
            loss_value = float(loss.detach().item())
            total_loss += loss_value
            should_print = (
                training
                and progress_prefix
                and (
                    is_tty
                    or idx % 20 == 0
                    or idx == len(samples)
                )
            )
            if should_print:
                if is_tty:
                    print(
                        f'\r{progress_prefix} loss {total_loss / idx:.4f} | '
                        f'step {idx:03d}/{len(samples)} [{step_time:.2f}s]',
                        end='',
                        flush=True,
                    )
                else:
                    print(
                        f'{progress_prefix} loss {total_loss / idx:.4f} | '
                        f'step {idx:03d}/{len(samples)} [{step_time:.2f}s]',
                        flush=True,
                    )
            all_logits.append(logits.detach().cpu())
            all_labels.append(sample.edge_labels.detach().cpu())

    if training and progress_prefix and is_tty:
        print()

    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)
    metrics = edge_f1(cat_logits, cat_labels)
    metrics['pred_pos_rate'] = float((torch.sigmoid(cat_logits) >= 0.5).float().mean().item())
    return total_loss / max(len(samples), 1), metrics, finalize_debug(epoch_debug)


@torch.no_grad()
def predict_logits_labels(
    model: MeshCNNSegmenter,
    samples: list[MeshCNNSample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits = []
    labels = []
    for sample in samples:
        logits.append(model(sample).detach().cpu())
        labels.append(sample.edge_labels.detach().cpu())
    return torch.cat(logits), torch.cat(labels)
