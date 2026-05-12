from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from models.common.gnn_registry import get_gnn_model
from models.common.gnn_train_data import (
    apply_runtime_feature_selection,
    dataset_metadata_summary,
    resolve_runtime_feature_selection,
    set_random_seeds,
)
from models.common.gnn_train_loop import (
    collect_logits_labels,
    confusion_counts,
    metric_line,
    print_threshold_sweep,
    run_epoch,
)
from models.common.gnn_train_runtime import build_runtime_config, logger_config, model_kwargs
from models.utils.dataset import (
    compute_pos_weight,
    filter_dataset_by_resolution,
    load_dataset,
    load_split_json_metadata,
    split_dataset,
)
from models.utils.experiment_log import ExperimentLogger
from models.utils.metrics import threshold_sweep


LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5


def train_gnn(args: argparse.Namespace) -> None:
    feature_selection = resolve_runtime_feature_selection(args)
    args.feature_group = feature_selection.feature_group
    args.in_dim = feature_selection.feature_count
    config = build_runtime_config(args)
    definition = get_gnn_model(config.model_name)

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

    model = definition.model_class(**model_kwargs(config)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    logger = ExperimentLogger(
        run_dir=args.run_dir,
        config=logger_config(
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
        train_loss, train_m = run_epoch(
            model, train, device, pos_weight, optimizer, config.focal_gamma
        )
        val_loss, val_m = run_epoch(model, val, device, pos_weight, focal_gamma=config.focal_gamma)
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
            f"prec {val_m['precision']:.4f}  "
            f"recall {val_m['recall']:.4f}  "
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
    test_loss, test_m = run_epoch(model, test, device, pos_weight, focal_gamma=config.focal_gamma)

    model.eval()
    val_logits_cat, val_labels_cat = collect_logits_labels(model, val, device)
    test_logits_cat, test_labels_cat = collect_logits_labels(model, test, device)

    val_sweep = threshold_sweep(val_logits_cat, val_labels_cat)
    test_sweep = threshold_sweep(test_logits_cat, test_labels_cat)
    best_t = val_sweep['best']['threshold']
    test_best_val_t_m = threshold_sweep(test_logits_cat, test_labels_cat, [best_t])['all'][0]

    logger.write_json('val_threshold_sweep.json', val_sweep)
    logger.write_json('test_threshold_sweep.json', test_sweep)

    print()
    print(metric_line('test @0.50', test_loss, test_m))
    print(metric_line(f'test @val-best {best_t:.2f}', None, test_best_val_t_m))

    print(f"\n{'-'*75}")
    print_threshold_sweep('threshold sweep (val):', val_sweep['all'], best_t, ' <-- best')
    print()
    print_threshold_sweep('threshold sweep (test):', test_sweep['all'], best_t, ' <-- best val')
    print(f"\noptimal threshold (by val F1): {best_t:.2f}")
    print(f"{'-'*75}")

    final_summary = {
        'seed': seed,
        'model_name': config.model_name,
        'hidden_dim': config.hidden_size,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
        'lr': config.lr,
        'split_json_in': str(args.split_json_in) if args.split_json_in else None,
        'split_json_out': str(args.split_json_out) if args.split_json_out else None,
        'best_validation_threshold': best_t,
        'test_metrics_threshold_0_5': test_m,
        'test_metrics_best_validation_threshold': test_best_val_t_m,
        'test_confusion_threshold_0_5': confusion_counts(test_m),
        'test_confusion_best_validation_threshold': confusion_counts(test_best_val_t_m),
        'resolution_tag': args.resolution_tag,
        'filtered_graph_count': filtered_graph_count,
        'dataset_metadata_summary': metadata_summary,
    }
    if config.model_name == 'gatv2':
        final_summary['heads'] = config.heads
    if config.model_name == 'graphsage':
        final_summary['skip_connections'] = config.skip_connections
        final_summary['aggr'] = config.aggr

    logger.finalize(
        test_metrics=test_m,
        best_epoch=best_epoch,
        extra_summary=final_summary,
    )
    logger.save()
    logger.plot()
