from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None

from models.meshcnn_full.mesh import load_meshcnn_dataset
from models.meshcnn_full.model import MeshCNNSegmenter
from models.meshcnn_full.training_data import (
    load_manifest,
    resolve_meshcnn_feature_selection,
    set_global_seed,
    slice_meshcnn_dataset_features,
    validate_dataset_tensors_cpu,
)
from models.meshcnn_full.training_loop import (
    format_debug_summary,
    log_debug_scalars,
    predict_logits_labels,
    run_epoch,
)
from models.utils.dataset import compute_pos_weight, load_split_json_metadata, split_dataset
from models.utils.metrics import edge_f1, threshold_sweep


def confusion_counts(metrics: dict[str, Any]) -> dict[str, int]:
    return {key: int(metrics[key]) for key in ('tp', 'fp', 'fn', 'tn') if key in metrics}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write('\n')


def parse_pool_ratios(value: str) -> tuple[float, float]:
    pool_ratios = tuple(float(item.strip()) for item in value.split(',') if item.strip())
    if len(pool_ratios) != 2:
        raise ValueError('--pool-ratios must contain exactly two comma-separated values')
    return pool_ratios


def resolve_seed(args) -> int:
    split_metadata = load_split_json_metadata(args.split_json_in) if args.split_json_in else {}
    seed_value = args.seed if args.seed is not None else split_metadata.get('seed')
    return int(seed_value) if seed_value is not None else 42


def prepare_training_dataset(args):
    dataset_path = Path(args.dataset)
    dataset = load_meshcnn_dataset(dataset_path)
    validate_dataset_tensors_cpu(dataset)
    print('[info] dataset tensors: cpu')
    manifest = load_manifest(dataset_path)
    source_feature_dim = int(manifest.get('feature_dim', dataset[0].edge_features.shape[1]))
    actual_source_channels = int(dataset[0].edge_features.shape[1])
    if source_feature_dim != actual_source_channels:
        print(
            f'[info] manifest feature_dim={source_feature_dim}, sample tensor has '
            f'{actual_source_channels}; using tensor shape for source validation'
        )
    selection = resolve_meshcnn_feature_selection(args)
    dataset, feature_metadata = slice_meshcnn_dataset_features(dataset, selection, manifest)
    feature_metadata['feature_dim'] = int(dataset[0].edge_features.shape[1])
    return dataset_path, dataset, feature_metadata


def split_training_dataset(dataset, dataset_path: Path, args, seed: int):
    train, val, test, split_info = split_dataset(
        dataset,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=seed,
        split_json_in=args.split_json_in,
        split_json_out=args.split_json_out,
        dataset_path=dataset_path,
        resolution_tag=args.resolution_tag,
    )
    if not train or not val or not test:
        raise ValueError('train/val/test split produced an empty split; use a larger dataset or adjust ratios')
    return train, val, test, split_info


def build_model_config(args, in_channels: int) -> dict[str, Any]:
    return {
        'in_channels': in_channels,
        'hidden_channels': args.hidden,
        'dropout': args.dropout,
        'pool_ratios': parse_pool_ratios(args.pool_ratios),
        'min_edges': args.min_edges,
    }


def build_training_config_payload(
    *,
    args,
    dataset_path: Path,
    model_config: dict[str, Any],
    feature_metadata: dict[str, Any],
    pos_weight: torch.Tensor,
    seed: int,
    train,
    val,
    test,
    split_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        'model': 'sparsemeshcnn',
        'dataset': str(dataset_path),
        'model_config': model_config,
        'feature_metadata': feature_metadata,
        'focal_gamma': args.focal_gamma,
        'pos_weight': float(pos_weight.item()),
        'grad_accum_steps': args.grad_accum_steps,
        'seed': seed,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'split_json_in': str(args.split_json_in) if args.split_json_in else None,
        'split_json_out': str(args.split_json_out) if args.split_json_out else None,
        'resolution_tag': args.resolution_tag,
        'train_graphs': len(train),
        'val_graphs': len(val),
        'test_graphs': len(test),
        'split': split_info,
    }


def print_run_header(
    *,
    device: torch.device,
    train,
    val,
    test,
    split_info: dict[str, Any],
    args,
    feature_metadata: dict[str, Any],
    in_channels: int,
    pos_weight: torch.Tensor,
) -> None:
    print(f'device: {device}')
    print(f'split: train {len(train)}, val {len(val)}, test {len(test)}')
    print(f'  train meshes: {split_info["train"]}')
    print(f'  val meshes:   {split_info["val"]}')
    print(f'  test meshes:  {split_info["test"]}')
    if args.split_json_out:
        print(f'split saved: {args.split_json_out}')
    print(f'features: {feature_metadata["feature_group"]} ({in_channels})')
    print(f'pos_weight: {pos_weight.item():.4f}')


def finalize_training(
    *,
    model: MeshCNNSegmenter,
    device: torch.device,
    pos_weight: torch.Tensor,
    args,
    test,
    val,
    best_path: Path,
    best_epoch: int,
    best_val_f1: float,
    feature_metadata: dict[str, Any],
    model_config: dict[str, Any],
    writer,
) -> None:
    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload['model_state'])
    test_loss, test_metrics, _ = run_epoch(
        model,
        test,
        device,
        pos_weight,
        args.focal_gamma,
    )
    val_logits, val_labels = predict_logits_labels(model, val, device)
    test_logits, test_labels = predict_logits_labels(model, test, device)
    val_sweep = threshold_sweep(val_logits, val_labels)
    test_sweep = threshold_sweep(test_logits, test_labels)
    best_validation_threshold = float(val_sweep['best']['threshold'])
    test_metrics_threshold_0_5 = edge_f1(test_logits, test_labels, threshold=0.5)
    test_metrics_best_validation_threshold = edge_f1(
        test_logits,
        test_labels,
        threshold=best_validation_threshold,
    )
    run_dir = best_path.parent
    write_json(run_dir / 'val_threshold_sweep.json', val_sweep)
    write_json(run_dir / 'test_threshold_sweep.json', test_sweep)
    summary = {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'best_validation_threshold': best_validation_threshold,
        'test_loss': test_loss,
        **{f'test_{key}': value for key, value in test_metrics.items() if isinstance(value, (int, float))},
        'test_metrics_threshold_0_5': test_metrics_threshold_0_5,
        'test_metrics_best_validation_threshold': test_metrics_best_validation_threshold,
        'test_confusion_threshold_0_5': confusion_counts(test_metrics_threshold_0_5),
        'test_confusion_best_validation_threshold': confusion_counts(test_metrics_best_validation_threshold),
        'feature_metadata': feature_metadata,
        'model_config': model_config,
    }
    write_json(run_dir / 'summary.json', summary)
    payload['best_validation_threshold'] = best_validation_threshold
    payload['val_threshold_sweep'] = val_sweep
    payload['test_threshold_sweep'] = test_sweep
    payload['test_metrics_threshold_0_5'] = test_metrics_threshold_0_5
    payload['test_metrics_best_validation_threshold'] = test_metrics_best_validation_threshold
    torch.save(payload, best_path)
    print(
        f'test | loss {test_loss:.4f} f1 {test_metrics["f1"]:.4f} '
        f'p {test_metrics["precision"]:.4f} r {test_metrics["recall"]:.4f}'
    )
    if writer is not None:
        writer.close()


def train_sparsemeshcnn(args) -> None:
    if not args.dataset:
        raise ValueError('--dataset is required')

    seed = resolve_seed(args)
    set_global_seed(seed)

    dataset_path, dataset, feature_metadata = prepare_training_dataset(args)
    train, val, test, split_info = split_training_dataset(dataset, dataset_path, args, seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = (
        torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
        if args.pos_weight is not None
        else compute_pos_weight(train).to(device)
    )
    in_channels = int(dataset[0].edge_features.shape[1])
    model_config = build_model_config(args, in_channels)
    model = MeshCNNSegmenter(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = build_training_config_payload(
        args=args,
        dataset_path=dataset_path,
        model_config=model_config,
        feature_metadata=feature_metadata,
        pos_weight=pos_weight,
        seed=seed,
        train=train,
        val=val,
        test=test,
        split_info=split_info,
    )
    write_json(run_dir / 'config.json', config_payload)
    writer = SummaryWriter(log_dir=str(run_dir)) if SummaryWriter is not None else None
    if writer is None:
        print('[info] tensorboard is unavailable; debug metrics will be shown in console only')

    print_run_header(
        device=device,
        train=train,
        val=val,
        test=test,
        split_info=split_info,
        args=args,
        feature_metadata=feature_metadata,
        in_channels=in_channels,
        pos_weight=pos_weight,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    metrics_log: list[dict[str, Any]] = []
    best_path = run_dir / 'best_model.pth'

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_metrics, train_debug = run_epoch(
            model,
            train,
            device,
            pos_weight,
            args.focal_gamma,
            optimizer=optimizer,
            grad_accum_steps=args.grad_accum_steps,
            progress_prefix=f'epoch {epoch:03d} | train',
        )
        val_loss, val_metrics, val_debug = run_epoch(
            model,
            val,
            device,
            pos_weight,
            args.focal_gamma,
        )
        scheduler.step(val_metrics['f1'])
        log_debug_scalars(writer, train_debug, epoch)
        log_debug_scalars(writer, val_debug, epoch, prefix='val')
        if writer is not None:
            writer.flush()
        elapsed = time.time() - epoch_start
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
        write_json(run_dir / 'metrics.json', metrics_log)
        print(
            f'epoch {epoch:03d} | train {train_loss:.4f} f1 {train_metrics["f1"]:.4f} | '
            f'val {val_loss:.4f} f1 {val_metrics["f1"]:.4f} '
            f'p {val_metrics["precision"]:.4f} r {val_metrics["recall"]:.4f} | {elapsed:.1f}s'
        )
        debug_summary = format_debug_summary(train_debug, train_metrics)
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

    finalize_training(
        model=model,
        device=device,
        pos_weight=pos_weight,
        args=args,
        test=test,
        val=val,
        best_path=best_path,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        feature_metadata=feature_metadata,
        model_config=model_config,
        writer=writer,
    )


def main(argv=None) -> None:
    from tools.run_training import run_single_model

    run_single_model(argv, 'sparsemeshcnn')


if __name__ == '__main__':
    main(sys.argv[1:])
