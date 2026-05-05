import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.baselines.registry import SUPPORTED_BASELINES, get_baseline
from models.common.baseline_train import train_baseline
from models.common.config import baseline_config


def _default_run_dir(model_name: str, timestamp: str) -> str:
    prefix = 'dual_graphsage' if model_name == 'graphsage' else model_name
    return f'runs/{prefix}_{timestamp}'


def _fill_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    definition = get_baseline(args.model)
    config = baseline_config(args.model, definition.default_config_overrides)

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
        'heads': config.heads,
        'aggr': config.aggr,
        'skip_connections': config.skip_connections,
    }
    for key, value in defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    shared_defaults = {
        'preset': 'extended',
        'val_ratio': 0.15,
        'test_ratio': 0.10,
        'seed': None,
        'split_json_in': None,
        'split_json_out': None,
        'resolution_tag': 'all',
        'feature_group': None,
        'enable_ao': False,
        'enable_dihedral': False,
        'enable_symmetry': False,
        'enable_density': False,
        'enable_thickness_sdf': False,
    }
    for key, value in shared_defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if not hasattr(args, 'run_dir') or args.run_dir is None:
        args.run_dir = _default_run_dir(args.model, datetime.now().strftime('%Y%m%d_%H%M%S'))
    return args


def build_parser(default_model: str = 'graphsage') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a baseline UV-seam model on the dual graph.')
    parser.add_argument('--model', choices=SUPPORTED_BASELINES, default=default_model)
    parser.add_argument('--dataset', default='dataset_dual.pt', help='path to dual dataset')
    parser.add_argument('--run-dir', default=None, help='experiment output dir')
    parser.add_argument('--preset', choices=['extended', 'paper'], default='extended',
                        help='training preset; paper sets GraphSeam-style GraphSAGE hyperparameters')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--heads', type=int, default=None)
    parser.add_argument('--num-layers', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None, help='early-stop patience')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for training and generated splits (default: 42 or split JSON seed)')
    parser.add_argument('--split-json-in', default=None, help='load train/val/test group ids from this JSON file')
    parser.add_argument('--split-json-out', default=None, help='save train/val/test group ids to this JSON file')
    parser.add_argument('--in-dim', type=int, default=None, help='dual node feature dim')
    parser.add_argument('--aggr', choices=['mean', 'lstm'], default=None,
                        help='GraphSAGE aggregation (default: mean)')
    parser.add_argument('--skip-connections', choices=['hidden', 'all', 'none'], default=None,
                        help='GraphSAGE residual mode (default: hidden)')
    parser.add_argument('--resolution-tag', default='all',
                        help='resolution selector: all, base, h, l, or a dataset-specific raw tag')
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='override pos_weight (default: auto-computed from dataset)')
    parser.add_argument('--focal-gamma', type=float, default=None,
                        help='focal loss gamma (0=plain BCE, 2=standard focal)')
    parser.add_argument('--feature-group', choices=['paper14', 'custom'], default=None,
                        help='feature bundle to train on (default: paper14)')
    parser.add_argument('--enable-ao', action='store_true',
                        help='enable AO endpoint features for --feature-group custom')
    parser.add_argument('--enable-dihedral', action='store_true',
                        help='enable signed dihedral for --feature-group custom')
    parser.add_argument('--enable-symmetry', action='store_true',
                        help='enable symmetry distance for --feature-group custom')
    parser.add_argument('--enable-density', action='store_true',
                        help='enable topology-local density features for --feature-group custom')
    parser.add_argument('--enable-thickness-sdf', action='store_true',
                        help='enable inward ray thickness for --feature-group custom')
    return parser


def parse_args(argv: list[str] | None = None, default_model: str = 'graphsage') -> argparse.Namespace:
    return _fill_model_defaults(build_parser(default_model).parse_args(argv))


def main(argv: list[str] | None = None, default_model: str = 'graphsage') -> None:
    args = parse_args(argv, default_model)
    train_baseline(args)


if __name__ == '__main__':
    main()
