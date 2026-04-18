from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.meshcnn_full.mesh import MeshCNNSample, load_meshcnn_dataset
from models.meshcnn_full.model import MeshCNNSegmenter
from models.utils.dataset import compute_pos_weight, load_split_json_metadata, split_dataset
from models.utils.losses import focal_bce_with_logits
from models.utils.metrics import edge_f1


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f'{dataset_path.stem}_manifest.json')


def _load_manifest(dataset_path: Path) -> dict[str, Any]:
    path = _manifest_path(dataset_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _feature_metadata(sample: MeshCNNSample, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        'feature_group': manifest.get('feature_group', sample.feature_group),
        'feature_preset': manifest.get('feature_preset', sample.feature_preset),
        'feature_names': manifest.get('feature_names', list(sample.feature_names)),
        'feature_flags': manifest.get('feature_flags', dict(sample.feature_flags)),
        'feature_dim': int(manifest.get('feature_dim', sample.in_channels)),
        'endpoint_order': manifest.get('endpoint_order', sample.endpoint_order),
        'density_config': manifest.get('density_config', sample.density_config),
        'label_source': manifest.get('label_source', sample.label_source),
        'sample_format': manifest.get('sample_format', 'meshcnn_full_v2'),
    }


def _validate_dataset_tensors_cpu(dataset: list[MeshCNNSample]) -> None:
    tensor_names = (
        'vertices',
        'faces',
        'unique_edges',
        'edge_features',
        'edge_labels',
    )
    for sample_idx, sample in enumerate(dataset[: min(8, len(dataset))]):
        for name in tensor_names:
            tensor = getattr(sample, name)
            if tensor.device.type != 'cpu':
                raise RuntimeError(
                    f'MeshCNNSample.{name} in sample {sample_idx} must be on CPU after dataset load, '
                    f'got {tensor.device}. Dataset loading must normalize samples to CPU.'
                )


def _loss_fn(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    loss_name: str,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_name == 'weighted-bce':
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    if loss_name == 'focal':
        return focal_bce_with_logits(logits, labels, pos_weight, gamma=focal_gamma)
    raise ValueError(f'unknown loss: {loss_name}')


def _safe_div(num: float, den: float, eps: float = 1e-8) -> float:
    return float(num) / float(den + eps)


def _mean_or_none(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _collect_pool_debug(model):
    out = []
    for module in model.modules():
        getter = getattr(module, 'get_last_debug', None)
        if getter is None:
            continue
        debug = getter()
        if debug:
            out.append(debug)
    return out


def _coerce_id_list(value) -> list[int] | None:
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


def _restored_orig_edge_ids(model) -> list[int] | None:
    for name in (
        'get_restored_orig_edge_ids',
        'get_last_restored_orig_edge_ids',
        'restored_orig_edge_ids',
        'last_restored_orig_edge_ids',
    ):
        value = getattr(model, name, None)
        ids = _coerce_id_list(value)
        if ids is not None:
            return ids
    return None


def _append_seam_enrichment(
    epoch_debug,
    layer_idx: int,
    labels: torch.Tensor,
    survivor_orig_edge_ids,
) -> None:
    survivor_ids = _coerce_id_list(survivor_orig_edge_ids)
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

    gt_retention = _safe_div(surviving_gt_seams, total_gt_seams)
    nonseam_retention = _safe_div(surviving_nonseams, total_nonseams)
    seam_enrichment = gt_retention / max(nonseam_retention, 1e-8)

    epoch_debug[f'pool/layer{layer_idx}_gt_seam_retention'].append(gt_retention)
    epoch_debug[f'pool/layer{layer_idx}_nonseam_retention'].append(nonseam_retention)
    epoch_debug[f'pool/layer{layer_idx}_seam_enrichment'].append(float(seam_enrichment))


def _append_debug_metrics(
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
        _append_seam_enrichment(
            epoch_debug,
            layer_idx,
            labels,
            debug.get('survivor_orig_edge_ids'),
        )
        for key in pool_metric_keys:
            epoch_debug[f'pool/layer{layer_idx}_{key}'].append(debug.get(key))

    expected_count = int(labels.numel())
    restored_ids = _restored_orig_edge_ids(model)
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


def _finalize_debug(epoch_debug) -> dict[str, float]:
    out: dict[str, float] = {}
    for tag, values in epoch_debug.items():
        value = _mean_or_none(values)
        if value is not None:
            out[tag] = float(value)
    return out


def _log_debug_scalars(writer, debug_metrics: dict[str, float], epoch: int, prefix: str = '') -> None:
    if writer is None:
        return
    for tag, value in debug_metrics.items():
        writer.add_scalar(f'{prefix}/{tag}' if prefix else tag, value, epoch)


def _format_debug_summary(debug_metrics: dict[str, float], metrics: dict[str, Any]) -> str:
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


def _run_epoch(
    model: MeshCNNSegmenter,
    samples: list[MeshCNNSample],
    device: torch.device,
    pos_weight: torch.Tensor,
    loss_name: str,
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
            pool_debug_list = _collect_pool_debug(model)
            _append_debug_metrics(epoch_debug, pool_debug_list, model, logits, labels)
            loss = _loss_fn(logits, labels, pos_weight, loss_name, focal_gamma)

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
    return total_loss / max(len(samples), 1), metrics, _finalize_debug(epoch_debug)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write('\n')


def main(argv: list[str] | None = None) -> None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser(description='Train isolated MeshCNN-full edge segmentation.')
    parser.add_argument('--dataset', default='dataset_meshcnn_full_paper14.pt')
    parser.add_argument('--run-dir', default=f'runs/meshcnn_full_{timestamp}')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--pool-ratios', default='0.85,0.75')
    parser.add_argument('--min-edges', type=int, default=32)
    parser.add_argument('--max-pool-collapses', type=int, default=2048)
    parser.add_argument('--loss', choices=('focal', 'weighted-bce'), default='focal')
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--pos-weight', type=float, default=None)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--group-mode', choices=('legacy', 'family'), default='family')
    parser.add_argument('--split-json-in', default=None)
    parser.add_argument('--split-json-out', default=None)
    args = parser.parse_args(argv)

    split_metadata = load_split_json_metadata(args.split_json_in) if args.split_json_in else {}
    seed_value = args.seed if args.seed is not None else split_metadata.get('seed')
    seed = int(seed_value) if seed_value is not None else 42
    set_global_seed(seed)

    dataset_path = Path(args.dataset)
    dataset = load_meshcnn_dataset(dataset_path)
    _validate_dataset_tensors_cpu(dataset)
    print('[info] dataset tensors: cpu')
    manifest = _load_manifest(dataset_path)
    feature_metadata = _feature_metadata(dataset[0], manifest)
    in_channels = int(feature_metadata.get('feature_dim') or dataset[0].edge_features.shape[1])
    actual_channels = int(dataset[0].edge_features.shape[1])
    if in_channels != actual_channels:
        print(f'[info] manifest feature_dim={in_channels}, sample tensor has {actual_channels}; using tensor shape')
        in_channels = actual_channels
        feature_metadata['feature_dim'] = actual_channels

    train, val, test, split_info = split_dataset(
        dataset,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=seed,
        group_mode=args.group_mode,
        split_json_in=args.split_json_in,
        split_json_out=args.split_json_out,
        dataset_path=dataset_path,
    )
    if not train or not val or not test:
        raise ValueError('train/val/test split produced an empty split; use a larger dataset or adjust ratios')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = (
        torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
        if args.pos_weight is not None
        else compute_pos_weight(train).to(device)
    )
    pool_ratios = tuple(float(item.strip()) for item in args.pool_ratios.split(',') if item.strip())
    if len(pool_ratios) != 2:
        parser.error('--pool-ratios must contain exactly two comma-separated values')

    model_config = {
        'in_channels': in_channels,
        'hidden_channels': args.hidden,
        'dropout': args.dropout,
        'pool_ratios': pool_ratios,
        'min_edges': args.min_edges,
        'max_pool_collapses': args.max_pool_collapses,
    }
    model = MeshCNNSegmenter(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        'model': 'meshcnn_full',
        'dataset': str(dataset_path),
        'model_config': model_config,
        'feature_metadata': feature_metadata,
        'loss': args.loss,
        'focal_gamma': args.focal_gamma,
        'pos_weight': float(pos_weight.item()),
        'grad_accum_steps': args.grad_accum_steps,
        'seed': seed,
        'group_mode': args.group_mode,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'split_json_in': str(args.split_json_in) if args.split_json_in else None,
        'split_json_out': str(args.split_json_out) if args.split_json_out else None,
        'train_graphs': len(train),
        'val_graphs': len(val),
        'test_graphs': len(test),
        'split': split_info,
    }
    _write_json(run_dir / 'config.json', config_payload)
    writer = SummaryWriter(log_dir=str(run_dir)) if SummaryWriter is not None else None
    if writer is None:
        print('[info] tensorboard is unavailable; debug metrics will be shown in console only')

    print(f'device: {device}')
    print(f'group_mode: {args.group_mode}')
    print(f'split: train {len(train)}, val {len(val)}, test {len(test)}')
    print(f'  train meshes: {split_info["train"]}')
    print(f'  val meshes:   {split_info["val"]}')
    print(f'  test meshes:  {split_info["test"]}')
    if args.split_json_out:
        print(f'split saved: {args.split_json_out}')
    print(f'features: {feature_metadata["feature_group"]} ({in_channels})')
    print(f'pos_weight: {pos_weight.item():.4f}')

    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    metrics_log: list[dict[str, Any]] = []
    best_path = run_dir / 'best_model.pth'

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics, train_debug = _run_epoch(
            model,
            train,
            device,
            pos_weight,
            args.loss,
            args.focal_gamma,
            optimizer=optimizer,
            grad_accum_steps=args.grad_accum_steps,
            progress_prefix=f'epoch {epoch:03d} | train',
        )
        val_loss, val_metrics, val_debug = _run_epoch(
            model,
            val,
            device,
            pos_weight,
            args.loss,
            args.focal_gamma,
        )
        scheduler.step(val_metrics['f1'])
        _log_debug_scalars(writer, train_debug, epoch)
        _log_debug_scalars(writer, val_debug, epoch, prefix='val')
        if writer is not None:
            writer.flush()
        elapsed = time.time() - t0
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_f1': train_metrics['f1'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_s': round(elapsed, 2),
        }
        metrics_log.append(row)
        _write_json(run_dir / 'metrics.json', metrics_log)
        print(
            f'epoch {epoch:03d} | train {train_loss:.4f} f1 {train_metrics["f1"]:.4f} | '
            f'val {val_loss:.4f} f1 {val_metrics["f1"]:.4f} '
            f'p {val_metrics["precision"]:.4f} r {val_metrics["recall"]:.4f} | {elapsed:.1f}s'
        )
        debug_summary = _format_debug_summary(train_debug, train_metrics)
        if debug_summary:
            print(f'  debug | {debug_summary}')

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'model_config': model_config,
                    'feature_metadata': feature_metadata,
                    'train_config': config_payload,
                    'best_epoch': best_epoch,
                    'best_val_f1': best_val_f1,
                },
                best_path,
            )
            print(f'  saved best -> {best_path}')
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f'early stopping at epoch {epoch}')
                break

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload['model_state'])
    test_loss, test_metrics, _test_debug = _run_epoch(
        model,
        test,
        device,
        pos_weight,
        args.loss,
        args.focal_gamma,
    )
    summary = {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_loss': test_loss,
        **{f'test_{key}': value for key, value in test_metrics.items() if isinstance(value, (int, float))},
    }
    _write_json(run_dir / 'summary.json', summary)
    print(
        f'test | loss {test_loss:.4f} f1 {test_metrics["f1"]:.4f} '
        f'p {test_metrics["precision"]:.4f} r {test_metrics["recall"]:.4f}'
    )
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
