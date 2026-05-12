import argparse
from datetime import datetime

try:
    from tools._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from models.common.gnn_registry import SUPPORTED_GNN_MODELS, get_gnn_model  # noqa: E402
from models.common.gnn_train import train_gnn  # noqa: E402
from models.meshcnn_full.train import train_sparsemeshcnn  # noqa: E402
from preprocessing.feature_registry import FEATURE_GROUP_NAMES  # noqa: E402


GNN_MODELS = tuple(SUPPORTED_GNN_MODELS)
TRAINING_MODELS = (*GNN_MODELS, 'sparsemeshcnn')


def _default_run_dir(model_name: str, timestamp: str) -> str:
    prefix = 'dual_graphsage' if model_name == 'graphsage' else model_name
    return f'runs/{prefix}_{timestamp}'


def _fill_gnn_defaults(args: argparse.Namespace) -> argparse.Namespace:
    definition = get_gnn_model(args.model)
    config = definition.train_config
    defaults = {
        'epochs': config.epochs,
        'lr': config.lr,
        'hidden': config.hidden_size,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
        'patience': config.patience,
        'in_dim': config.in_dim,
        'pos_weight': config.pos_weight,
        'focal_gamma': config.focal_gamma,
        'dataset': 'dataset_dual.pt',
        'feature_group': None,
    }
    if args.model == 'gatv2':
        defaults['heads'] = config.heads
    if args.model == 'graphsage':
        defaults['skip_connections'] = config.skip_connections
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def _fill_sparsemeshcnn_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        'dataset': 'dataset_sparsemeshcnn_paper14.pt',
        'epochs': 100,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'hidden': 64,
        'dropout': 0.2,
        'pool_ratios': '0.85,0.75',
        'min_edges': 32,
        'focal_gamma': 2.0,
        'pos_weight': None,
        'grad_accum_steps': 1,
        'patience': 50,
        'feature_group': 'paper14',
    }
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def fill_training_defaults(args: argparse.Namespace) -> argparse.Namespace:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.model in GNN_MODELS:
        args = _fill_gnn_defaults(args)
    elif args.model == 'sparsemeshcnn':
        args = _fill_sparsemeshcnn_defaults(args)
    else:
        raise ValueError(f'unsupported model: {args.model}')

    shared_defaults = {
        'val_ratio': 0.15,
        'test_ratio': 0.10,
        'seed': None,
        'split_json_in': None,
        'split_json_out': None,
        'resolution_tag': 'all',
        'enable_ao': False,
        'enable_dihedral': False,
        'enable_symmetry': False,
        'enable_density': False,
        'enable_thickness_sdf': False,
        'mean_debug': False,
    }
    for key, value in shared_defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    if getattr(args, 'run_dir', None) is None:
        args.run_dir = _default_run_dir(args.model, timestamp)
    return args


def _default_model(model_choices: tuple[str, ...]) -> str:
    if not model_choices:
        raise ValueError('model_choices must not be empty')
    return model_choices[0]


def build_parser(
    model_choices: tuple[str, ...] = TRAINING_MODELS,
    description: str = 'Train a UV-seam prediction model.',
    include_model_arg: bool = True,
) -> argparse.ArgumentParser:
    include_gnn_options = any(model in GNN_MODELS for model in model_choices)
    include_graphsage_options = 'graphsage' in model_choices
    include_gatv2_options = 'gatv2' in model_choices
    include_sparse_options = 'sparsemeshcnn' in model_choices
    parser = argparse.ArgumentParser(description=description)
    if include_model_arg:
        parser.add_argument('--model', choices=model_choices, default=_default_model(model_choices))
    parser.add_argument('--dataset', default=None, help='path to the model-specific dataset')
    parser.add_argument('--run-dir', default=None, help='experiment output directory')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--split-json-in', default=None)
    parser.add_argument('--split-json-out', default=None)
    parser.add_argument('--resolution-tag', default='all')
    parser.add_argument('--pos-weight', type=float, default=None)
    parser.add_argument('--focal-gamma', type=float, default=None)
    parser.add_argument('--feature-group', choices=FEATURE_GROUP_NAMES, default=None)
    parser.add_argument('--enable-ao', action='store_true')
    parser.add_argument('--enable-dihedral', action='store_true')
    parser.add_argument('--enable-symmetry', action='store_true')
    parser.add_argument('--enable-density', action='store_true')
    parser.add_argument('--enable-thickness-sdf', action='store_true')
    parser.add_argument('--mean_debug', action='store_true')

    if include_gnn_options:
        parser.add_argument('--num-layers', type=int, default=None)
        parser.add_argument('--in-dim', type=int, default=None)
    if include_gatv2_options:
        parser.add_argument('--heads', type=int, default=None)
    if include_graphsage_options:
        parser.add_argument('--skip-connections', choices=['hidden', 'all', 'none'], default=None)

    parser.add_argument('--weight-decay', type=float, default=None)

    if include_sparse_options:
        parser.add_argument('--pool-ratios', default=None)
        parser.add_argument('--min-edges', type=int, default=None)
        parser.add_argument('--grad-accum-steps', type=int, default=None)
    return parser


def parse_args(
    argv: list[str] | None = None,
    model_choices: tuple[str, ...] = TRAINING_MODELS,
    description: str = 'Train a UV-seam prediction model.',
    include_model_arg: bool = True,
) -> argparse.Namespace:
    args = build_parser(model_choices, description, include_model_arg).parse_args(argv)
    return _normalize_namespace(args, model_choices)


def train_graph_model(args: argparse.Namespace) -> None:
    train_gnn(args)


def train_model(args: argparse.Namespace) -> None:
    if args.model in GNN_MODELS:
        train_graph_model(args)
        return
    if args.model == 'sparsemeshcnn':
        train_sparsemeshcnn(args)
        return
    raise ValueError(f'unsupported model: {args.model}')


def _normalize_namespace(
    args: argparse.Namespace,
    model_choices: tuple[str, ...],
) -> argparse.Namespace:
    if getattr(args, 'model', None) is None:
        args.model = _default_model(model_choices)
    if args.model not in model_choices:
        raise ValueError(f'unsupported model for this entrypoint: {args.model}')
    return fill_training_defaults(args)


def main(
    argv: argparse.Namespace | list[str] | None = None,
    model_choices: tuple[str, ...] = TRAINING_MODELS,
    description: str = 'Train a UV-seam prediction model.',
    include_model_arg: bool = True,
) -> None:
    if isinstance(argv, argparse.Namespace):
        train_model(_normalize_namespace(argv, model_choices))
        return
    train_model(parse_args(argv, model_choices, description, include_model_arg))


def run_single_model(
    argv: argparse.Namespace | list[str] | None,
    model_name: str,
    description: str = 'Train a UV-seam prediction model.',
) -> None:
    main(argv, model_choices=(model_name,), description=description, include_model_arg=False)


if __name__ == '__main__':
    main()
