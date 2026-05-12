from __future__ import annotations

from pathlib import Path

import torch

from models.common.gnn_registry import get_gnn_model
from models.common.gnn_train_data import apply_runtime_feature_selection
from models.common.gnn_train_loop import collect_logits_labels
from models.common.gnn_train_runtime import model_kwargs
from models.common.gnn_config import replace_config
from models.utils.dataset import filter_dataset_by_resolution, load_dataset, load_split_json_metadata, split_dataset
from preprocessing.feature_registry import resolve_feature_selection
from tools.utils.ablation_specs import EXPERIMENT_SPECS
from tools.utils.ablation_splits import split_path_for_seed
from tools.utils.json_io import read_json
from tools.utils.reeval_common import SavedRun, metric_delta, split_identity
from tools.utils.reeval_thresholds import exact_validation_threshold, metrics_at_threshold, threshold_table


def infer_experiment(run_dir: Path, runs_root: Path) -> str | None:
    try:
        parts = run_dir.resolve().relative_to(runs_root.resolve()).parts
    except ValueError:
        parts = run_dir.parts
    if 'experiments' in parts:
        index = parts.index('experiments')
        if index + 1 < len(parts):
            return parts[index + 1]
    if run_dir.parent.name in EXPERIMENT_SPECS:
        return run_dir.parent.name
    return None


def parse_seed(run_dir: Path, config: dict, summary: dict) -> int:
    for source in (config, summary):
        value = source.get('seed')
        if value is not None:
            return int(value)
    if run_dir.name.startswith('seed_'):
        return int(run_dir.name.split('_', 1)[1])
    raise ValueError(f'{run_dir} is missing seed metadata')


def experiment_summary(run_dir: Path, experiment: str | None) -> dict:
    if experiment is None:
        return {}
    summary_path = run_dir.parent / 'summary.json'
    if not summary_path.exists():
        return {}
    payload = read_json(summary_path)
    return payload if isinstance(payload, dict) else {}


def select_dataset_path(
    *,
    config: dict,
    experiment_summary: dict,
    gnn_dataset: str | None,
) -> Path:
    if gnn_dataset:
        return Path(gnn_dataset)
    dataset = config.get('dataset') or experiment_summary.get('dataset')
    if not dataset:
        raise ValueError('missing dataset path metadata; provide --gnn-dataset')
    return Path(dataset)


def require_config(config: dict, run_dir: Path) -> None:
    required = ['model_name', 'in_dim', 'hidden_dim', 'num_layers', 'dropout', 'dataset', 'seed']
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f'{run_dir / "config.json"} missing required field(s): {", ".join(missing)}')
    if config.get('model_name') == 'graphsage':
        if config.get('skip_connections') is None:
            raise ValueError(f'{run_dir / "config.json"} missing required field: skip_connections')


def discover_saved_runs(args) -> list[SavedRun]:
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise ValueError(f'--runs-root does not exist: {runs_root}')
    splits_dir = Path(args.splits_dir)
    targets: list[SavedRun] = []
    requested_experiments = set(args.experiments or [])
    requested_seeds = {int(seed) for seed in args.seeds} if args.seeds else None

    for checkpoint_path in sorted(runs_root.rglob('best_model.pth')):
        run_dir = checkpoint_path.parent
        experiment = infer_experiment(run_dir, runs_root)
        if requested_experiments and experiment not in requested_experiments:
            continue

        config_path = run_dir / 'config.json'
        summary_path = run_dir / 'summary.json'
        if not config_path.exists():
            raise ValueError(f'{run_dir} has best_model.pth but no config.json')
        if not summary_path.exists():
            raise ValueError(f'{run_dir} has best_model.pth but no summary.json')

        config = read_json(config_path)
        summary = read_json(summary_path)
        if not isinstance(config, dict) or not isinstance(summary, dict):
            raise ValueError(f'{run_dir} config.json and summary.json must contain objects')
        require_config(config, run_dir)

        seed = parse_seed(run_dir, config, summary)
        if requested_seeds is not None and seed not in requested_seeds:
            continue

        run_experiment_summary = experiment_summary(run_dir, experiment)
        split_path = split_path_for_seed(splits_dir, seed)
        if not split_path.exists():
            raise ValueError(f'missing frozen split JSON for seed {seed}: {split_path}')
        dataset_path = select_dataset_path(
            config=config,
            experiment_summary=run_experiment_summary,
            gnn_dataset=args.gnn_dataset,
        )

        targets.append(SavedRun(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            summary_path=summary_path,
            split_path=split_path,
            dataset_path=dataset_path,
            experiment=experiment,
            seed=seed,
            config=config,
            summary=summary,
        ))

    if not targets:
        raise ValueError('no matching runs with best_model.pth were found')
    return targets


def feature_selection_from_config(config: dict):
    flags = config.get('feature_flags') or {}
    return resolve_feature_selection(
        config.get('feature_group'),
        enable_ao=bool(flags.get('ao', False)),
        enable_dihedral=bool(flags.get('signed_dihedral', False)),
        enable_symmetry=bool(flags.get('symmetry', False)),
        enable_density=bool(flags.get('density', False)),
        enable_thickness_sdf=bool(flags.get('thickness_sdf', False)),
    )


def runtime_config_from_saved(config: dict):
    definition = get_gnn_model(config['model_name'])
    return replace_config(
        definition.train_config,
        hidden_size=config.get('hidden_dim'),
        num_layers=config.get('num_layers'),
        in_dim=config.get('in_dim'),
        dropout=config.get('dropout'),
        lr=config.get('lr'),
        pos_weight=config.get('pos_weight'),
        focal_gamma=config.get('focal_gamma'),
        patience=config.get('patience'),
        heads=config.get('heads'),
        skip_connections=config.get('skip_connections'),
        aggr=config.get('aggr'),
    )


def load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise ValueError(f'checkpoint must contain a state dict: {path}')
    return checkpoint


def old_validation_best_metrics(run_dir: Path) -> dict | None:
    path = run_dir / 'val_threshold_sweep.json'
    if not path.exists():
        return None
    payload = read_json(path)
    if isinstance(payload, dict):
        return payload.get('best')
    return None


def evaluate_saved_run(target: SavedRun, *, device: torch.device, report_grid: list[float]) -> dict:
    config = target.config
    selection = feature_selection_from_config(config)
    runtime_config = runtime_config_from_saved(config)
    definition = get_gnn_model(runtime_config.model_name)

    dataset = load_dataset(target.dataset_path)
    resolution_tag = config.get('resolution_tag', 'all')
    dataset = filter_dataset_by_resolution(dataset, resolution_tag)
    dataset = apply_runtime_feature_selection(dataset, selection)

    split_payload = load_split_json_metadata(target.split_path)
    split_dataset_path = split_payload.get('dataset_path') or None
    _, val, test, split_info = split_dataset(
        dataset,
        seed=target.seed,
        split_json_in=target.split_path,
        dataset_path=split_dataset_path,
        resolution_tag=resolution_tag,
    )

    model = definition.model_class(**model_kwargs(runtime_config)).to(device)
    model.load_state_dict(load_state_dict(target.checkpoint_path, device))
    model.eval()

    val_logits, val_labels = collect_logits_labels(model, val, device)
    test_logits, test_labels = collect_logits_labels(model, test, device)
    val_probs = torch.sigmoid(val_logits)
    test_probs = torch.sigmoid(test_logits)

    exact = exact_validation_threshold(val_probs, val_labels)
    exact_threshold = float(exact['threshold'])
    old_validation_best = old_validation_best_metrics(target.run_dir)
    old_threshold = target.summary.get('best_validation_threshold')
    if old_threshold is None and old_validation_best:
        old_threshold = old_validation_best.get('threshold')

    val_metrics = {
        'threshold_0_5': metrics_at_threshold(val_probs, val_labels, 0.5),
        'exact_val_best': exact['metrics'],
    }
    test_metrics = {
        'threshold_0_5': metrics_at_threshold(test_probs, test_labels, 0.5),
        'exact_val_best': metrics_at_threshold(test_probs, test_labels, exact_threshold),
    }
    if old_threshold is not None:
        old_threshold = float(old_threshold)
        val_metrics['old_val_best'] = metrics_at_threshold(val_probs, val_labels, old_threshold)
        test_metrics['old_val_best'] = metrics_at_threshold(test_probs, test_labels, old_threshold)
    else:
        val_metrics['old_val_best'] = None
        test_metrics['old_val_best'] = None

    old_stored = {
        'validation_val_best': old_validation_best,
        'test_val_best': target.summary.get('test_metrics_best_validation_threshold'),
        'test_0_5': target.summary.get('test_metrics_threshold_0_5'),
    }

    payload = {
        'status': 'completed',
        'run_identity': {
            'experiment': target.experiment,
            'seed': target.seed,
            'run_dir': str(target.run_dir),
        },
        'checkpoint_path': str(target.checkpoint_path),
        'split_path': str(target.split_path),
        'dataset_path': str(target.dataset_path),
        'model_family': {
            'model_name': runtime_config.model_name,
            'display_name': definition.display_name,
        },
        'feature_selection': {
            'feature_group': selection.feature_group,
            'feature_flags': selection.feature_flags.as_dict(),
            'feature_names': list(selection.feature_names),
        },
        'split': {
            'seed': split_info.get('seed'),
            'resolution_tag': split_info.get('resolution_tag'),
            'val_graphs': len(val),
            'test_graphs': len(test),
        },
        'threshold_search': {
            'method': 'exact_validation_f1_over_score_breakpoints',
            'candidate_source': exact['candidate_source'],
            'candidate_count': exact['candidate_count'],
            'tie_breaking': exact['tie_breaking'],
            'dense_report_grid': report_grid,
        },
        'old_threshold': old_threshold,
        'exact_validation_optimal_threshold': exact_threshold,
        'metrics': {
            'validation': val_metrics,
            'test': test_metrics,
        },
        'dense_grid': {
            'validation': threshold_table(val_probs, val_labels, report_grid),
            'test': threshold_table(test_probs, test_labels, report_grid),
        },
        'old_stored_metrics': old_stored,
        'comparison': {
            'delta_vs_old_stored_val_best': {
                'validation': metric_delta(val_metrics['exact_val_best'], old_stored['validation_val_best']),
                'test': metric_delta(test_metrics['exact_val_best'], old_stored['test_val_best']),
            },
            'delta_vs_0_5': {
                'validation': metric_delta(val_metrics['exact_val_best'], val_metrics['threshold_0_5']),
                'test': metric_delta(test_metrics['exact_val_best'], test_metrics['threshold_0_5']),
            },
        },
    }
    payload['split_identity'] = split_identity(payload)
    return payload


def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('--device cuda requested but CUDA is not available')
    return torch.device(name)
