import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.dual_graphsage.model import DualGraphSAGE
from models.utils.dataset import (
    compute_pos_weight,
    filter_dataset_by_resolution,
    load_dataset,
    load_split_json_metadata,
    split_dataset,
)
from models.utils.experiment_log import ExperimentLogger
from models.utils.losses import focal_bce_with_logits, seam_loss_with_connectivity
from models.utils.metrics import edge_f1, threshold_sweep


METADATA_KEYS = ('label_source', 'feature_preset', 'endpoint_order', 'weld_mode')


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metadata_value(data: Data, key: str):
    try:
        value = getattr(data, key)
        if value not in (None, ''):
            return value
    except AttributeError:
        pass

    for container_key in ('metadata', 'meta', 'dataset_metadata'):
        try:
            container = getattr(data, container_key)
        except AttributeError:
            continue
        if isinstance(container, dict) and key in container and container[key] not in (None, ''):
            return container[key]
        if hasattr(container, key):
            value = getattr(container, key)
            if value not in (None, ''):
                return value
    return None


def dataset_metadata_summary(dataset: list[Data]) -> dict:
    summary: dict = {'graph_count': len(dataset)}
    for key in METADATA_KEYS:
        values = []
        missing = 0
        for data in dataset:
            value = _metadata_value(data, key)
            if value is None:
                missing += 1
            else:
                values.append(str(value))

        if values:
            unique_values = sorted(set(values))
            summary[key] = unique_values[0] if len(unique_values) == 1 else unique_values
        if missing and (values or missing != len(dataset)):
            summary[f'{key}_missing'] = missing

    feature_dims = []
    for data in dataset:
        x = getattr(data, 'x', None)
        if x is not None and getattr(x, 'ndim', 0) == 2:
            feature_dims.append(int(x.shape[1]))
    if feature_dims:
        unique_dims = sorted(set(feature_dims))
        summary['x_feature_dim'] = unique_dims[0] if len(unique_dims) == 1 else unique_dims

    return summary


def validate_strict_paper_protocol(args: argparse.Namespace, dataset: list[Data]) -> None:
    failures = []
    if args.preset != 'paper':
        failures.append("preset must be 'paper'")
    if not getattr(args, 'resolution_tag', None):
        failures.append('resolution_tag must be set')
    if args.in_dim != 14:
        failures.append('in_dim must be 14')
    if args.aggr != 'lstm':
        failures.append("aggr must be 'lstm'")
    if args.skip_connections != 'all':
        failures.append("skip_connections must be 'all'")

    for key, expected in (('label_source', 'exact_obj'), ('feature_preset', 'paper14')):
        values = [_metadata_value(data, key) for data in dataset]
        observed = sorted({str(value) for value in values if value not in (None, '')})
        missing = sum(1 for value in values if value in (None, ''))
        if missing or observed != [expected]:
            detail = f"observed={observed or 'none'}"
            if missing:
                detail += f", missing={missing}"
            failures.append(f"dataset {key} must be {expected!r} ({detail})")

    if failures:
        raise ValueError('strict paper protocol failed: ' + '; '.join(failures))


def _metric_line(label: str, loss: float | None, metrics: dict) -> str:
    loss_part = f"loss {loss:.4f}  " if loss is not None else ''
    return (
        f"{label} | {loss_part}f1 {metrics['f1']:.4f}  "
        f"prec {metrics['precision']:.4f}  rec {metrics['recall']:.4f}  "
        f"tpr {metrics['tpr']:.4f}  fpr {metrics['fpr']:.4f}  acc {metrics['accuracy']:.4f}"
    )


def _confusion_counts(metrics: dict) -> dict:
    return {key: int(metrics[key]) for key in ('tp', 'fp', 'fn', 'tn')}


def _run_epoch(
    model: DualGraphSAGE,
    graphs: list[Data],
    device: torch.device,
    pos_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    lambda_conn: float = 0.0,
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

            if lambda_conn > 0.0:
                loss = seam_loss_with_connectivity(logits, y, edge_index, pos_weight, lambda_conn, focal_gamma)
            else:
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


def main(args: argparse.Namespace) -> None:
    split_metadata = load_split_json_metadata(args.split_json_in) if args.split_json_in else {}
    effective_seed = args.seed if args.seed is not None else int(split_metadata.get('seed', 42))
    effective_group_mode = args.group_mode or split_metadata.get('group_mode', 'legacy')

    set_random_seeds(effective_seed)

    if args.preset == 'paper':
        args.lr = 5e-4
        args.hidden = 64
        args.num_layers = 3
        args.pos_weight = 100.0
        args.focal_gamma = 0.0
        args.patience = 50
        args.in_dim = 14
        args.aggr = 'lstm'
        args.skip_connections = 'all'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    dataset = load_dataset(args.dataset)
    dataset = filter_dataset_by_resolution(dataset, args.resolution_tag)
    filtered_graph_count = len(dataset)
    print(f"resolution selector: {args.resolution_tag} ({filtered_graph_count} graph(s))")

    if args.strict_paper_protocol:
        validate_strict_paper_protocol(args, dataset)

    metadata_summary = dataset_metadata_summary(dataset)

    train, val, test, split_info = split_dataset(
        dataset,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=effective_seed,
        group_mode=effective_group_mode,
        split_json_in=args.split_json_in,
        split_json_out=args.split_json_out,
        dataset_path=args.dataset,
        resolution_tag=args.resolution_tag,
    )
    print(f"split - train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"  train meshes: {split_info['train']}")
    print(f"  val meshes:   {split_info['val']}")
    print(f"  test meshes:  {split_info['test']}")
    if args.split_json_out:
        print(f"split saved: {args.split_json_out}")

    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32).to(device)
        print(f"pos_weight: {pos_weight.item():.4f} (manual override)")
    else:
        pos_weight = compute_pos_weight(train).to(device)
        print(f"pos_weight: {pos_weight.item():.4f} (auto-computed)")

    model = DualGraphSAGE(
        in_dim=args.in_dim,
        hidden_dim=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        aggr=args.aggr,
        skip_connections=args.skip_connections,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    logger = ExperimentLogger(
        run_dir=args.run_dir,
        config={
            'model': 'DualGraphSAGE',
            'in_dim': args.in_dim,
            'hidden_dim': args.hidden,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'preset': args.preset,
            'aggr': args.aggr,
            'skip_connections': args.skip_connections,
            'lr': args.lr,
            'lambda_conn': args.lambda_conn,
            'focal_gamma': args.focal_gamma,
            'patience': args.patience,
            'dataset': args.dataset,
            'resolution_tag': args.resolution_tag,
            'resolution_selector': args.resolution_tag,
            'filtered_graph_count': filtered_graph_count,
            'seed': effective_seed,
            'group_mode': effective_group_mode,
            'split_json_in': str(args.split_json_in) if args.split_json_in else None,
            'split_json_out': str(args.split_json_out) if args.split_json_out else None,
            'train_graphs': len(train),
            'val_graphs': len(val),
            'test_graphs': len(test),
            'pos_weight': pos_weight.item(),
            'split': split_info,
            'dataset_metadata_summary': metadata_summary,
        },
    )
    logger.log_class_balance(train, val, test)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_ctr = 0
    save_path = Path(args.run_dir) / 'best_model.pth'

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = _run_epoch(
            model, train, device, pos_weight, optimizer, args.lambda_conn, args.focal_gamma
        )
        val_loss, val_m = _run_epoch(model, val, device, pos_weight, focal_gamma=args.focal_gamma)
        epoch_time = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_m['f1'])

        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=current_lr,
            epoch_time_s=round(epoch_time, 2),
            train_f1=train_m['f1'],
            train_precision=train_m['precision'],
            train_recall=train_m['recall'],
            val_f1=val_m['f1'],
            val_precision=val_m['precision'],
            val_recall=val_m['recall'],
            train_accuracy=train_m['accuracy'],
            train_fpr=train_m['fpr'],
            train_tpr=train_m['tpr'],
            val_accuracy=val_m['accuracy'],
            val_fpr=val_m['fpr'],
            val_tpr=val_m['tpr'],
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}  f1 {train_m['f1']:.4f} | "
            f"val loss {val_loss:.4f}  f1 {val_m['f1']:.4f}  "
            f"prec {val_m['precision']:.4f}  rec {val_m['recall']:.4f}  "
            f"tpr {val_m['tpr']:.4f}  fpr {val_m['fpr']:.4f}  acc {val_m['accuracy']:.4f}  "
            f"[{epoch_time:.1f}s]"
        )

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best model (val F1 = {best_val_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    print(f"\nloading best weights from {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_m = _run_epoch(model, test, device, pos_weight, focal_gamma=args.focal_gamma)

    # Threshold sweep on val (select) and test (report)
    model.eval()
    val_logits, val_labels = [], []
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for data in val:
            val_logits.append(model(data.x.to(device), data.edge_index.to(device)).cpu())
            val_labels.append(data.y.cpu())
        for data in test:
            test_logits_list.append(model(data.x.to(device), data.edge_index.to(device)).cpu())
            test_labels_list.append(data.y.cpu())
    val_logits_cat = torch.cat(val_logits)
    val_labels_cat = torch.cat(val_labels)
    test_logits_cat = torch.cat(test_logits_list)
    test_labels_cat = torch.cat(test_labels_list)

    val_sweep = threshold_sweep(val_logits_cat, val_labels_cat)
    test_sweep = threshold_sweep(test_logits_cat, test_labels_cat)
    best_t = val_sweep['best']['threshold']
    test_best_val_t_m = edge_f1(test_logits_cat, test_labels_cat, threshold=best_t)

    logger.write_json('val_threshold_sweep.json', val_sweep)
    logger.write_json('test_threshold_sweep.json', test_sweep)

    print()
    print(_metric_line('test @0.50', test_loss, test_m))
    print(_metric_line(f'test @val-best {best_t:.2f}', None, test_best_val_t_m))

    print(f"\n{'-'*75}")
    print("threshold sweep (val):")
    print(f"  {'t':>5s}  {'P':>7s}  {'R':>7s}  {'F1':>7s}  {'FPR':>7s}")
    for r in val_sweep['all']:
        marker = ' <-- best' if r['threshold'] == best_t else ''
        print(f"  {r['threshold']:>5.2f}  {r['precision']:>7.4f}  {r['recall']:>7.4f}  {r['f1']:>7.4f}  {r['fpr']:>7.4f}{marker}")
    print("\nthreshold sweep (test):")
    print(f"  {'t':>5s}  {'P':>7s}  {'R':>7s}  {'F1':>7s}  {'FPR':>7s}")
    for r in test_sweep['all']:
        marker = ' <-- best val' if r['threshold'] == best_t else ''
        print(f"  {r['threshold']:>5.2f}  {r['precision']:>7.4f}  {r['recall']:>7.4f}  {r['f1']:>7.4f}  {r['fpr']:>7.4f}{marker}")
    print(f"\noptimal threshold (by val F1): {best_t:.2f}")
    print(f"{'-'*75}")

    logger.finalize(
        test_metrics=test_m,
        best_epoch=best_epoch,
        extra_summary={
            'seed': effective_seed,
            'group_mode': effective_group_mode,
            'split_json_in': str(args.split_json_in) if args.split_json_in else None,
            'split_json_out': str(args.split_json_out) if args.split_json_out else None,
            'best_validation_threshold': best_t,
            'test_metrics_threshold_0_5': test_m,
            'test_metrics_best_validation_threshold': test_best_val_t_m,
            'test_confusion_threshold_0_5': _confusion_counts(test_m),
            'test_confusion_best_validation_threshold': _confusion_counts(test_best_val_t_m),
            'resolution_tag': args.resolution_tag,
            'resolution_selector': args.resolution_tag,
            'filtered_graph_count': filtered_graph_count,
            'preset': args.preset,
            'dataset_metadata_summary': metadata_summary,
        },
    )
    logger.save()
    logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DualGraphSAGE on dual graph for UV-seam prediction.")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument('--dataset', default='dataset_dual.pt', help='path to dual dataset')
    parser.add_argument('--run-dir', default=f'runs/dual_graphsage_{timestamp}', help='experiment output dir')
    parser.add_argument('--preset', choices=['extended', 'paper'], default='extended',
                        help='training preset; paper sets GraphSeam-style hyperparameters')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lambda-conn', type=float, default=0.0,
                        help='connectivity penalty weight (0 = disabled, try 0.1)')
    parser.add_argument('--patience', type=int, default=15, help='early-stop patience')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for training and generated splits (default: 42 or split JSON seed)')
    parser.add_argument('--group-mode', choices=['legacy', 'family'], default=None,
                        help='grouping mode for generated or loaded splits (default: legacy or split JSON value)')
    parser.add_argument('--split-json-in', default=None, help='load train/val/test group ids from this JSON file')
    parser.add_argument('--split-json-out', default=None, help='save train/val/test group ids to this JSON file')
    parser.add_argument('--in-dim', type=int, default=18, help='dual node feature dim (default: 18)')
    parser.add_argument('--aggr', choices=['mean', 'lstm'], default='mean',
                        help='GraphSAGE aggregation (default: mean)')
    parser.add_argument('--skip-connections', choices=['hidden', 'all', 'none'], default='hidden',
                        help='Residual mode (default preserves current behavior)')
    parser.add_argument('--resolution-tag', default='all',
                        help='resolution selector: all, base, h, l, or a dataset-specific raw tag')
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='override pos_weight (default: auto-computed from dataset)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='focal loss gamma (0=plain BCE, 2=standard focal)')
    parser.add_argument('--strict-paper-protocol', action='store_true',
                        help='fail unless dataset and options match the paper-faithful GraphSeam protocol')

    main(parser.parse_args())
