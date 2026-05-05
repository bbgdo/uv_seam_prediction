import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from models.baselines.registry import get_baseline
from models.common.config import BaselineConfig, baseline_config, replace_config
from models.utils.dataset import (
    compute_pos_weight,
    filter_dataset_by_resolution,
    load_dataset,
    load_split_json_metadata,
    split_dataset,
)
from models.utils.experiment_log import ExperimentLogger
from models.utils.losses import focal_bce_with_logits
from models.utils.metrics import RECALL_TPR_LABEL, edge_f1, threshold_sweep
from preprocessing.feature_registry import PAPER14_FEATURE_NAMES, ResolvedFeatureSet, resolve_feature_selection


METADATA_KEYS = (
    'label_source',
    'feature_preset',
    'feature_group',
    'feature_names',
    'feature_flags',
    'density_config',
    'endpoint_order',
    'weld_mode',
)


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


def resolve_runtime_feature_selection(args: argparse.Namespace) -> ResolvedFeatureSet:
    feature_group = getattr(args, 'feature_group', None)
    if feature_group is None:
        feature_group = 'paper14'

    return resolve_feature_selection(
        feature_group,
        enable_ao=bool(getattr(args, 'enable_ao', False)),
        enable_dihedral=bool(getattr(args, 'enable_dihedral', False)),
        enable_symmetry=bool(getattr(args, 'enable_symmetry', False)),
        enable_density=bool(getattr(args, 'enable_density', False)),
        enable_thickness_sdf=bool(getattr(args, 'enable_thickness_sdf', False)),
    )


def _coerce_feature_names(value) -> list[str] | None:
    if value in (None, ''):
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def _feature_names_from_saved_paper14_dim(data: Data) -> list[str] | None:
    preset = _metadata_value(data, 'feature_preset')
    if preset == 'paper14' and getattr(data.x, 'shape', (0, 0))[1] == 14:
        return list(PAPER14_FEATURE_NAMES)
    return None


def apply_runtime_feature_selection(dataset: list[Data], selection: ResolvedFeatureSet) -> list[Data]:
    requested = list(selection.feature_names)
    for graph_idx, data in enumerate(dataset):
        feature_names = _coerce_feature_names(_metadata_value(data, 'feature_names'))
        if feature_names is None:
            feature_names = _feature_names_from_saved_paper14_dim(data)

        current_dim = int(data.x.shape[1])
        if feature_names is None:
            if current_dim == selection.feature_count and selection.feature_group == 'paper14':
                continue
            raise ValueError(
                f"dataset graph {graph_idx} is missing feature_names metadata; "
                f"cannot select requested features {requested}"
            )
        if len(feature_names) != current_dim:
            raise ValueError(
                f"dataset graph {graph_idx} feature_names length {len(feature_names)} "
                f"does not match x feature dim {current_dim}"
            )

        missing = [name for name in requested if name not in feature_names]
        if missing:
            raise ValueError(
                f"dataset graph {graph_idx} is missing requested feature(s): {missing}; "
                f"available feature_names={feature_names}"
            )

        if feature_names == requested:
            continue

        indices = [feature_names.index(name) for name in requested]
        data.x = data.x[:, indices]
        data.feature_names = requested
        data.feature_group = selection.feature_group
        data.feature_preset = selection.feature_preset
        data.feature_flags = selection.feature_flags.as_dict()
        if selection.density_config is not None:
            data.density_config = dict(selection.density_config)

    return dataset



def apply_paper_preset(args: argparse.Namespace) -> None:
    if args.model != 'graphsage' or args.preset != 'paper':
        return
    args.lr = 5e-4
    args.hidden = 64
    args.num_layers = 3
    args.pos_weight = 100.0
    args.focal_gamma = 0.0
    args.patience = 50
    args.in_dim = 14
    args.aggr = 'lstm'
    args.skip_connections = 'all'


def build_runtime_config(args: argparse.Namespace) -> BaselineConfig:
    definition = get_baseline(args.model)
    config = baseline_config(args.model, definition.default_config_overrides)
    return replace_config(
        config,
        hidden_size=args.hidden,
        num_layers=args.num_layers,
        lr=args.lr,
        pos_weight=args.pos_weight,
        focal_gamma=args.focal_gamma,
        epochs=args.epochs,
        patience=args.patience,
        in_dim=args.in_dim,
        dropout=args.dropout,
        heads=args.heads,
        aggr=args.aggr,
        skip_connections=args.skip_connections,
    )


def _metric_line(label: str, loss: float | None, metrics: dict) -> str:
    loss_part = f"loss {loss:.4f}  " if loss is not None else ''
    return (
        f"{label} | {loss_part}f1 {metrics['f1']:.4f}  "
        f"prec {metrics['precision']:.4f}  {RECALL_TPR_LABEL} {metrics['recall']:.4f}  "
        f"fpr {metrics['fpr']:.4f}  acc {metrics['accuracy']:.4f}"
    )


def _confusion_counts(metrics: dict) -> dict:
    return {key: int(metrics[key]) for key in ('tp', 'fp', 'fn', 'tn')}


def _run_epoch(
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


def _collect_logits_labels(
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


def _model_kwargs(config: BaselineConfig) -> dict:
    kwargs = {
        'in_dim': config.in_dim,
        'hidden_dim': config.hidden_size,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
    }
    if config.model_name == 'graphsage':
        kwargs.update({
            'aggr': config.aggr,
            'skip_connections': config.skip_connections,
        })
    elif config.model_name == 'gatv2':
        kwargs['heads'] = config.heads
    return kwargs


def _logger_config(
    args: argparse.Namespace,
    config: BaselineConfig,
    display_name: str,
    pos_weight: torch.Tensor,
    split_info: dict,
    metadata_summary: dict,
    filtered_graph_count: int,
    seed: int | None,
    split_sizes: tuple[int, int, int],
) -> dict:
    train_count, val_count, test_count = split_sizes
    payload = {
        'model': display_name,
        'model_name': config.model_name,
        'hidden': config.hidden_size,
        'in_dim': config.in_dim,
        'hidden_dim': config.hidden_size,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
        'lr': config.lr,
        'focal_gamma': config.focal_gamma,
        'patience': config.patience,
        'dataset': args.dataset,
        'feature_group': getattr(args, 'feature_group', None),
        'feature_flags': {
            'ao': bool(getattr(args, 'enable_ao', False)),
            'signed_dihedral': bool(getattr(args, 'enable_dihedral', False)),
            'symmetry': bool(getattr(args, 'enable_symmetry', False)),
            'density': bool(getattr(args, 'enable_density', False)),
            'thickness_sdf': bool(getattr(args, 'enable_thickness_sdf', False)),
        },
        'resolution_tag': args.resolution_tag,
        'resolution_selector': args.resolution_tag,
        'filtered_graph_count': filtered_graph_count,
        'seed': seed,
        'split_json_in': str(args.split_json_in) if args.split_json_in else None,
        'split_json_out': str(args.split_json_out) if args.split_json_out else None,
        'train_graphs': train_count,
        'val_graphs': val_count,
        'test_graphs': test_count,
        'pos_weight': pos_weight.item(),
        'split': split_info,
        'dataset_metadata_summary': metadata_summary,
    }
    if config.model_name == 'graphsage':
        payload.update({
            'aggr': config.aggr,
            'skip_connections': config.skip_connections,
        })
    elif config.model_name == 'gatv2':
        payload['heads'] = config.heads
    return payload


def _print_threshold_sweep(title: str, rows: list[dict], best_t: float, marker_label: str) -> None:
    print(title)
    print(f"  {'t':>5s}  {'P':>7s}  {RECALL_TPR_LABEL:>8s}  {'F1':>7s}  {'FPR':>7s}")
    for row in rows:
        marker = marker_label if row['threshold'] == best_t else ''
        print(
            f"  {row['threshold']:>5.2f}  {row['precision']:>7.4f}  "
            f"{row['recall']:>8.4f}  {row['f1']:>7.4f}  {row['fpr']:>7.4f}{marker}"
        )


def train_baseline(args: argparse.Namespace) -> None:
    apply_paper_preset(args)
    feature_selection = resolve_runtime_feature_selection(args)
    args.feature_group = feature_selection.feature_group
    args.in_dim = feature_selection.feature_count
    config = build_runtime_config(args)
    definition = get_baseline(config.model_name)

    split_metadata = load_split_json_metadata(args.split_json_in) if args.split_json_in else {}
    seed_value = args.seed if args.seed is not None else split_metadata.get('seed')
    if seed_value is None and config.model_name == 'graphsage':
        seed_value = 42
    seed = int(seed_value) if seed_value is not None else None
    split_seed = seed if seed is not None else 42

    if seed is not None:
        set_random_seeds(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    print(
        f"features: {feature_selection.feature_group} "
        f"({feature_selection.feature_count}) [{', '.join(feature_selection.feature_names)}]"
    )

    dataset = load_dataset(args.dataset)
    dataset = filter_dataset_by_resolution(dataset, args.resolution_tag)
    filtered_graph_count = len(dataset)
    print(f"resolution selector: {args.resolution_tag} ({filtered_graph_count} graph(s))")

    dataset = apply_runtime_feature_selection(dataset, feature_selection)
    metadata_summary = dataset_metadata_summary(dataset)
    train, val, test, split_info = split_dataset(
        dataset,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=split_seed,
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

    if config.pos_weight is not None:
        pos_weight = torch.tensor([config.pos_weight], dtype=torch.float32).to(device)
        print(f"pos_weight: {pos_weight.item():.4f} (manual override)")
    else:
        pos_weight = compute_pos_weight(train).to(device)
        print(f"pos_weight: {pos_weight.item():.4f} (auto-computed)")

    model = definition.model_class(**_model_kwargs(config)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.scheduler_factor, patience=config.scheduler_patience
    )

    logger = ExperimentLogger(
        run_dir=args.run_dir,
        config=_logger_config(
            args,
            config,
            definition.display_name,
            pos_weight,
            split_info,
            metadata_summary,
            filtered_graph_count,
            seed,
            (len(train), len(val), len(test)),
        ),
    )
    logger.log_class_balance(train, val, test)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_ctr = 0
    save_path = Path(args.run_dir) / 'best_model.pth'

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_loss, train_m = _run_epoch(
            model, train, device, pos_weight, optimizer, config.focal_gamma
        )
        val_loss, val_m = _run_epoch(model, val, device, pos_weight, focal_gamma=config.focal_gamma)
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
            f"prec {val_m['precision']:.4f}  {RECALL_TPR_LABEL} {val_m['recall']:.4f}  "
            f"fpr {val_m['fpr']:.4f}  acc {val_m['accuracy']:.4f}  "
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
            if patience_ctr >= config.patience:
                print(f"early stopping at epoch {epoch} (no improvement for {config.patience} epochs).")
                break

    print(f"\nloading best weights from {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_m = _run_epoch(model, test, device, pos_weight, focal_gamma=config.focal_gamma)

    model.eval()
    val_logits_cat, val_labels_cat = _collect_logits_labels(model, val, device)
    test_logits_cat, test_labels_cat = _collect_logits_labels(model, test, device)

    val_sweep = threshold_sweep(val_logits_cat, val_labels_cat, config.threshold_values)
    test_sweep = threshold_sweep(test_logits_cat, test_labels_cat, config.threshold_values)
    best_t = val_sweep['best']['threshold']
    test_best_val_t_m = edge_f1(test_logits_cat, test_labels_cat, threshold=best_t)

    logger.write_json('val_threshold_sweep.json', val_sweep)
    logger.write_json('test_threshold_sweep.json', test_sweep)

    print()
    print(_metric_line('test @0.50', test_loss, test_m))
    print(_metric_line(f'test @val-best {best_t:.2f}', None, test_best_val_t_m))

    print(f"\n{'-'*75}")
    _print_threshold_sweep('threshold sweep (val):', val_sweep['all'], best_t, ' <-- best')
    print()
    _print_threshold_sweep('threshold sweep (test):', test_sweep['all'], best_t, ' <-- best val')
    print(f"\noptimal threshold (by val F1): {best_t:.2f}")
    print(f"{'-'*75}")

    logger.finalize(
        test_metrics=test_m,
        best_epoch=best_epoch,
        extra_summary={
            'seed': seed,
            'model_name': config.model_name,
            'hidden': config.hidden_size,
            'hidden_dim': config.hidden_size,
            'heads': config.heads if config.model_name == 'gatv2' else None,
            'num_layers': config.num_layers,
            'dropout': config.dropout,
            'lr': config.lr,
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
            'dataset_metadata_summary': metadata_summary,
        },
    )
    logger.save()
    logger.plot()
