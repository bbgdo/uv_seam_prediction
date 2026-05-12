from __future__ import annotations

import argparse

import torch

from models.common.gnn_registry import get_gnn_model
from models.common.gnn_config import GNNTrainConfig, replace_config


def build_runtime_config(args: argparse.Namespace) -> GNNTrainConfig:
    definition = get_gnn_model(args.model)
    config = definition.train_config
    aggr = 'mean' if args.model == 'graphsage' and getattr(args, 'mean_debug', False) else None
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
        weight_decay=getattr(args, 'weight_decay', None),
        heads=getattr(args, 'heads', None),
        skip_connections=getattr(args, 'skip_connections', None),
        aggr=aggr,
    )


def model_kwargs(config: GNNTrainConfig) -> dict:
    kwargs = {
        'in_dim': config.in_dim,
        'hidden_dim': config.hidden_size,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
    }
    if config.model_name == 'graphsage':
        kwargs.update({
            'skip_connections': config.skip_connections,
            'aggr': config.aggr,
        })
    elif config.model_name == 'gatv2':
        kwargs['heads'] = config.heads
    return kwargs


def logger_config(
    args: argparse.Namespace,
    config: GNNTrainConfig,
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
            'skip_connections': config.skip_connections,
            'aggr': config.aggr,
        })
    elif config.model_name == 'gatv2':
        payload['heads'] = config.heads
    return payload
