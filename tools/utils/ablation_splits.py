from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from models.utils.dataset import load_split_json_metadata, split_dataset

from .ablation_specs import BASELINE_EXPERIMENT


def split_path_for_seed(splits_dir: Path, seed: int) -> Path:
    return splits_dir / f'seed_{seed}.json'


def split_json_for_seed(args: argparse.Namespace, seed: int) -> Path:
    split_json_in = getattr(args, 'split_json_in', None)
    if split_json_in:
        return Path(split_json_in)
    return split_path_for_seed(Path(args.splits_dir), seed)


def _validate_split_metadata(payload: dict[str, Any], split_json: Path, args: argparse.Namespace, seed: int) -> None:
    missing = [
        key for key in ('train_group_ids', 'val_group_ids', 'test_group_ids', 'seed', 'resolution_tag')
        if key not in payload
    ]
    if missing:
        raise ValueError(f"{split_json} missing required field(s): {', '.join(sorted(missing))}")
    if int(payload['seed']) != int(seed):
        raise ValueError(f"{split_json} seed={payload['seed']!r} does not match requested seed={seed}")
    if payload.get('resolution_tag') != args.resolution_tag:
        raise ValueError(
            f"{split_json} resolution_tag={payload.get('resolution_tag')!r} "
            f"does not match requested resolution_tag={args.resolution_tag!r}"
        )
    if payload.get('dataset_path') not in (None, ''):
        raise ValueError(
            f"{split_json} is tied to dataset_path={payload.get('dataset_path')!r}; "
            "ablation splits must be dataset-agnostic so paper14 and custom runs reuse identical groups"
        )


def generate_split_files(
    *,
    source_dataset: list,
    splits_dir: Path,
    seeds: list[int],
    resolution_tag: str,
    val_ratio: float,
    test_ratio: float,
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        split_path = split_path_for_seed(splits_dir, seed)
        if split_path.exists():
            continue
        split_dataset(
            source_dataset,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_json_out=split_path,
            dataset_path=None,
            resolution_tag=resolution_tag,
        )


def validate_split_files(args: argparse.Namespace, datasets: dict[str, list]) -> None:
    for seed in args.seeds:
        split_json = split_json_for_seed(args, seed)
        if not split_json.exists():
            raise ValueError(f"missing split JSON for seed {seed}: {split_json}")
        payload = load_split_json_metadata(split_json)
        _validate_split_metadata(payload, split_json, args, seed)
        for dataset in datasets.values():
            split_dataset(
                dataset,
                split_json_in=split_json,
                dataset_path=None,
                resolution_tag=args.resolution_tag,
            )


def resolve_control14_run_dir(path: str | None, model: str, seed: int, *, allow_direct_run: bool) -> Path | None:
    if not path:
        return None
    root = Path(path)
    if allow_direct_run and (root / 'summary.json').exists():
        return root
    candidates = (
        root / f'seed_{seed}',
        root / model / 'experiments' / BASELINE_EXPERIMENT / f'seed_{seed}',
        root / 'experiments' / BASELINE_EXPERIMENT / f'seed_{seed}',
        root / BASELINE_EXPERIMENT / f'seed_{seed}',
    )
    for candidate in candidates:
        if (candidate / 'summary.json').exists():
            return candidate
    raise ValueError(f'control14 run dir for seed {seed} must contain summary.json: {root}')
