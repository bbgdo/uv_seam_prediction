from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from .ablation_datasets import get_gnn_dataset_arg
from .ablation_reports import collect_success_record, failure_record
from .ablation_splits import resolve_control14_run_dir, split_json_for_seed
from .ablation_specs import (
    ABLATION_MODELS,
    BASELINE_EXPERIMENT,
    ExperimentSpec,
    experiment_feature_label,
    is_meshcnn_model,
)


TRAINING_SCRIPT = Path('tools') / 'run_training.py'


def build_train_command(
    *,
    spec: ExperimentSpec,
    dataset: str | None,
    meshcnn_dataset: str | None = None,
    run_dir: Path,
    split_json: Path,
    seed: int,
    resolution_tag: str,
    epochs: int,
    patience: int,
    model: str = 'graphsage',
    mean_debug: bool = False,
) -> list[str]:
    if model not in ABLATION_MODELS:
        choices = ', '.join(ABLATION_MODELS)
        raise ValueError(f"unsupported ablation model {model!r}; choose one of: {choices}")
    if is_meshcnn_model(model):
        if not meshcnn_dataset:
            raise ValueError(f'{spec.name} requires a MeshCNN dataset')
        command = [
            sys.executable,
            str(TRAINING_SCRIPT),
            '--model',
            model,
            '--dataset',
            meshcnn_dataset,
            '--run-dir',
            str(run_dir),
            '--epochs',
            str(epochs),
            '--patience',
            str(patience),
            '--seed',
            str(seed),
            '--split-json-in',
            str(split_json),
            '--resolution-tag',
            resolution_tag,
            '--feature-group',
            spec.feature_group,
        ]
        if spec.enable_ao:
            command.append('--enable-ao')
        if spec.enable_dihedral:
            command.append('--enable-dihedral')
        if spec.enable_symmetry:
            command.append('--enable-symmetry')
        if spec.enable_density:
            command.append('--enable-density')
        if spec.enable_thickness_sdf:
            command.append('--enable-thickness-sdf')
        if mean_debug:
            command.append('--mean_debug')
        return command

    if not dataset:
        raise ValueError(f'{spec.name} requires a dataset')

    command = [
        sys.executable,
        str(TRAINING_SCRIPT),
        '--model',
        model,
        '--dataset',
        dataset,
        '--run-dir',
        str(run_dir),
        '--resolution-tag',
        resolution_tag,
        '--seed',
        str(seed),
        '--split-json-in',
        str(split_json),
        '--epochs',
        str(epochs),
        '--patience',
        str(patience),
        '--feature-group',
        spec.feature_group,
    ]
    if spec.enable_ao:
        command.append('--enable-ao')
    if spec.enable_dihedral:
        command.append('--enable-dihedral')
    if spec.enable_symmetry:
        command.append('--enable-symmetry')
    if spec.enable_density:
        command.append('--enable-density')
    if spec.enable_thickness_sdf:
        command.append('--enable-thickness-sdf')
    if mean_debug:
        command.append('--mean_debug')
    return command


def run_experiment(
    *,
    args: argparse.Namespace,
    spec: ExperimentSpec,
    runner=subprocess.run,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model = getattr(args, 'model', 'graphsage')
    experiment_dir = Path(args.output_root) / model / 'experiments' / spec.name
    for seed in args.seeds:
        split_json = split_json_for_seed(args, seed)
        external_baseline = (
            resolve_control14_run_dir(
                getattr(args, 'control14_run_dir', None),
                model,
                seed,
                allow_direct_run=len(args.seeds) == 1,
            )
            if spec.name == BASELINE_EXPERIMENT
            else None
        )
        if external_baseline is not None:
            print(
                f"{model}: {spec.phase}/{spec.name} seed {seed} "
                f"features={experiment_feature_label(spec)} using baseline run {external_baseline}"
            )
            records.append(collect_success_record(seed, external_baseline, split_json))
            continue

        run_dir = experiment_dir / f'seed_{seed}'
        run_dir.mkdir(parents=True, exist_ok=True)
        command = build_train_command(
            spec=spec,
            dataset=get_gnn_dataset_arg(args),
            meshcnn_dataset=getattr(args, 'meshcnn_dataset', None),
            run_dir=run_dir,
            split_json=split_json,
            seed=seed,
            resolution_tag=args.resolution_tag,
            epochs=args.epochs,
            patience=args.patience,
            model=model,
            mean_debug=getattr(args, 'mean_debug', False),
        )

        print(
            f"{model}: {spec.phase}/{spec.name} seed {seed} "
            f"features={experiment_feature_label(spec)} run_dir={run_dir}"
        )
        try:
            runner(command, check=True)
            records.append(collect_success_record(seed, run_dir, split_json))
        except subprocess.CalledProcessError as exc:
            records.append(failure_record(seed, run_dir, split_json, f'train runner exited with {exc.returncode}'))
            if not args.keep_going:
                return records
        except Exception as exc:
            records.append(failure_record(seed, run_dir, split_json, str(exc)))
            if not args.keep_going:
                return records
    return records
