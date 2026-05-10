from __future__ import annotations

import argparse

import torch

from models.baselines.registry import get_baseline
from models.common.config import BaselineConfig, baseline_config, replace_config


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
        weight_decay=getattr(args, 'weight_decay', None),
        heads=args.heads,
        aggr=args.aggr,
        skip_connections=args.skip_connections,
    )


def model_kwargs(config: BaselineConfig) -> dict:
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


def logger_config(
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
