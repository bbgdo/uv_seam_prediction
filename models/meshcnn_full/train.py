from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.meshcnn_full.mesh import MeshCNNSample, load_meshcnn_dataset
from models.meshcnn_full.model import MeshCNNSegmenter
from models.utils.dataset import compute_pos_weight, split_dataset
from models.utils.losses import focal_bce_with_logits
from models.utils.metrics import edge_f1


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


def _run_epoch(
    model: MeshCNNSegmenter,
    samples: list[MeshCNNSample],
    device: torch.device,
    pos_weight: torch.Tensor,
    loss_name: str,
    focal_gamma: float,
    optimizer: torch.optim.Optimizer | None = None,
    grad_accum_steps: int = 1,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    if training:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for idx, sample in enumerate(samples, start=1):
            labels = sample.edge_labels.to(device)
            logits = model(sample)
            loss = _loss_fn(logits, labels, pos_weight, loss_name, focal_gamma)

            if training:
                (loss / max(grad_accum_steps, 1)).backward()
                if idx % grad_accum_steps == 0 or idx == len(samples):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().item())
            all_logits.append(logits.detach().cpu())
            all_labels.append(sample.edge_labels.detach().cpu())

    metrics = edge_f1(torch.cat(all_logits), torch.cat(all_labels))
    return total_loss / max(len(samples), 1), metrics


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
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    dataset = load_meshcnn_dataset(dataset_path)
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
        seed=args.seed,
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
        'split': split_info,
    }
    _write_json(run_dir / 'config.json', config_payload)

    print(f'device: {device}')
    print(f'split: train {len(train)}, val {len(val)}, test {len(test)}')
    print(f'features: {feature_metadata["feature_group"]} ({in_channels})')
    print(f'pos_weight: {pos_weight.item():.4f}')

    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    metrics_log: list[dict[str, Any]] = []
    best_path = run_dir / 'best_model.pth'

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = _run_epoch(
            model,
            train,
            device,
            pos_weight,
            args.loss,
            args.focal_gamma,
            optimizer=optimizer,
            grad_accum_steps=args.grad_accum_steps,
        )
        val_loss, val_metrics = _run_epoch(
            model,
            val,
            device,
            pos_weight,
            args.loss,
            args.focal_gamma,
        )
        scheduler.step(val_metrics['f1'])
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
    test_loss, test_metrics = _run_epoch(model, test, device, pos_weight, args.loss, args.focal_gamma)
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


if __name__ == '__main__':
    main()
