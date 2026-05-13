from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from models.meshcnn_full.mesh import load_meshcnn_dataset
from models.meshcnn_full.model import MeshCNNSegmenter
from models.meshcnn_full.training_data import load_manifest, slice_meshcnn_dataset_features
from models.meshcnn_full.training_loop import predict_logits_labels
from models.utils.dataset import filter_dataset_by_resolution, load_split_json_metadata, split_dataset
from preprocessing.feature_registry import resolve_feature_selection
from tools.utils.ablation_splits import split_path_for_seed
from tools.utils.json_io import read_json
from tools.utils.reeval_common import SavedRun
from tools.utils.reeval_payloads import build_reevaluation_payload
from tools.utils.reeval_gnn import infer_experiment, parse_seed


def require_meshcnn_config(config: dict[str, Any], run_dir: Path) -> None:
    if config.get('model') != 'sparsemeshcnn':
        raise ValueError(f'{run_dir / "config.json"} is not a sparsemeshcnn config')
    required = ['dataset', 'feature_metadata', 'model_config', 'seed']
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f'{run_dir / "config.json"} missing required field(s): {", ".join(missing)}')


def discover_saved_meshcnn_runs(args: Namespace) -> list[SavedRun]:
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise ValueError(f'--runs-root does not exist: {runs_root}')
    splits_dir = Path(args.splits_dir)
    requested_experiments = set(args.experiments or [])
    requested_seeds = {int(seed) for seed in args.seeds} if args.seeds else None
    targets: list[SavedRun] = []

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
        require_meshcnn_config(config, run_dir)

        seed = parse_seed(run_dir, config, summary)
        if requested_seeds is not None and seed not in requested_seeds:
            continue

        split_path = split_path_for_seed(splits_dir, seed)
        if not split_path.exists():
            raise ValueError(f'missing frozen split JSON for seed {seed}: {split_path}')
        dataset_path = Path(args.meshcnn_dataset or config['dataset'])
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
        raise ValueError('no matching SparseMeshCNN runs with best_model.pth were found')
    return targets


def meshcnn_feature_selection(config: dict[str, Any]):
    metadata = config.get('feature_metadata') or {}
    flags = metadata.get('feature_flags') or {}
    return resolve_feature_selection(
        metadata.get('feature_group'),
        enable_ao=bool(flags.get('ao', False)),
        enable_dihedral=bool(flags.get('signed_dihedral', False)),
        enable_symmetry=bool(flags.get('symmetry', False)),
        enable_density=bool(flags.get('density', False)),
        enable_thickness_sdf=bool(flags.get('thickness_sdf', False)),
    )


def load_meshcnn_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get('model_state'), dict):
        return checkpoint['model_state']
    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError(f'checkpoint must contain a SparseMeshCNN state dict: {path}')


def meshcnn_model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(config.get('model_config') or {})
    allowed = {'in_channels', 'hidden_channels', 'dropout', 'pool_ratios', 'min_edges'}
    kwargs = {key: raw[key] for key in allowed if key in raw}
    if 'pool_ratios' in kwargs:
        kwargs['pool_ratios'] = tuple(float(value) for value in kwargs['pool_ratios'])
    return kwargs


def old_meshcnn_validation_best_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / 'val_threshold_sweep.json'
    if not path.exists():
        return None
    payload = read_json(path)
    if isinstance(payload, dict):
        return payload.get('best')
    return None


def evaluate_saved_meshcnn_run(
    target: SavedRun,
    *,
    device: torch.device,
    report_grid: list[float],
    threshold_decimals: int | None = None,
) -> dict[str, Any]:
    config = target.config
    selection = meshcnn_feature_selection(config)
    dataset = load_meshcnn_dataset(target.dataset_path)
    resolution_tag = config.get('resolution_tag', 'all')
    dataset = filter_dataset_by_resolution(dataset, resolution_tag)
    dataset, feature_metadata = slice_meshcnn_dataset_features(
        dataset,
        selection,
        load_manifest(target.dataset_path),
    )

    split_payload = load_split_json_metadata(target.split_path)
    split_dataset_path = split_payload.get('dataset_path') or None
    _, val, test, split_info = split_dataset(
        dataset,
        seed=target.seed,
        split_json_in=target.split_path,
        dataset_path=split_dataset_path,
        resolution_tag=resolution_tag,
    )

    model = MeshCNNSegmenter(**meshcnn_model_kwargs(config)).to(device)
    model.load_state_dict(load_meshcnn_state(target.checkpoint_path, device))
    model.eval()

    val_logits, val_labels = predict_logits_labels(model, val, device)
    test_logits, test_labels = predict_logits_labels(model, test, device)

    return build_reevaluation_payload(
        target=target,
        model_name='sparsemeshcnn',
        display_name='SparseMeshCNN',
        feature_selection={
            'feature_group': selection.feature_group,
            'feature_flags': selection.feature_flags.as_dict(),
            'feature_names': list(selection.feature_names),
            'feature_metadata': feature_metadata,
        },
        split_info=split_info,
        val_graphs=len(val),
        test_graphs=len(test),
        val_logits=val_logits,
        val_labels=val_labels,
        test_logits=test_logits,
        test_labels=test_labels,
        report_grid=report_grid,
        threshold_decimals=threshold_decimals,
        old_validation_best=old_meshcnn_validation_best_metrics(target.run_dir),
    )
