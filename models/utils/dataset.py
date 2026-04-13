import json
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.utils.filename_parsing import FilenameParseConfig, legacy_base_name, parse_mesh_name


SPLIT_GROUP_KEYS = {
    'train': 'train_group_ids',
    'val': 'val_group_ids',
    'test': 'test_group_ids',
}


def load_dataset(path: str | Path) -> list[Data]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    dataset = torch.load(path, weights_only=False)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"expected a non-empty list of Data objects, got: {type(dataset)}")

    return dataset


def infer_resolution_selector(path_or_name: str | Path) -> str:
    stem = legacy_base_name(path_or_name).lower()
    if stem.endswith('_h'):
        return 'h'
    if stem.endswith('_l'):
        return 'l'

    parsed_tag = parse_mesh_name(path_or_name).resolution_tag
    if parsed_tag:
        return parsed_tag

    return 'base'


def available_resolution_selectors(dataset: list[Data]) -> list[str]:
    return sorted({infer_resolution_selector(getattr(d, 'file_path', '')) for d in dataset})


def filter_dataset_by_resolution(dataset: list[Data], resolution_tag: str | None) -> list[Data]:
    if not resolution_tag or resolution_tag == 'all':
        return dataset

    filtered = [
        d for d in dataset
        if infer_resolution_selector(getattr(d, 'file_path', '')) == resolution_tag
    ]
    if not filtered:
        available = ', '.join(available_resolution_selectors(dataset)) or 'none'
        raise ValueError(
            f"no graphs matched resolution selector {resolution_tag!r}; available selectors: {available}"
        )
    return filtered


def _normalize_dataset_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve(strict=False))


def _group_name(d: Data, group_mode: str, filename_config: FilenameParseConfig | None = None) -> str:
    name = Path(getattr(d, 'file_path', '')).stem
    if not name:
        return str(id(d))
    if group_mode == 'family':
        return parse_mesh_name(name, filename_config).family_id
    return legacy_base_name(name)


def _group_dataset(
    dataset: list[Data],
    group_mode: str,
    filename_config: FilenameParseConfig | None = None,
) -> dict[str, list[Data]]:
    if group_mode not in {'legacy', 'family'}:
        raise ValueError(f"group_mode must be 'legacy' or 'family', got: {group_mode}")

    groups: dict[str, list[Data]] = {}
    for d in dataset:
        groups.setdefault(_group_name(d, group_mode, filename_config), []).append(d)
    return groups


def load_split_json_metadata(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"split JSON must contain an object: {path}")
    return payload


def _split_info(
    train_keys: list[str],
    val_keys: list[str],
    test_keys: list[str],
    seed: int,
    group_mode: str,
    dataset_path: str | Path | None,
    resolution_tag: str | None,
) -> dict:
    return {
        'train': sorted(train_keys),
        'val': sorted(val_keys),
        'test': sorted(test_keys),
        'seed': seed,
        'group_mode': group_mode,
        'dataset_path': _normalize_dataset_path(dataset_path),
        'resolution_tag': resolution_tag,
    }


def _split_json_payload(split_info: dict) -> dict:
    return {
        'train_group_ids': split_info['train'],
        'val_group_ids': split_info['val'],
        'test_group_ids': split_info['test'],
        'seed': split_info.get('seed'),
        'group_mode': split_info.get('group_mode'),
        'dataset_path': split_info.get('dataset_path'),
        'resolution_tag': split_info.get('resolution_tag'),
    }


def save_split_json(path: str | Path, split_info: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_split_json_payload(split_info), f, indent=2)


def _validate_split_payload(
    payload: dict,
    groups: dict[str, list[Data]],
    group_mode: str,
    dataset_path: str | Path | None,
    resolution_tag: str | None,
) -> dict:
    missing_keys = [key for key in SPLIT_GROUP_KEYS.values() if key not in payload]
    missing_keys.extend(key for key in ('seed', 'group_mode', 'dataset_path', 'resolution_tag') if key not in payload)
    if missing_keys:
        raise ValueError(f"split JSON missing required field(s): {', '.join(sorted(missing_keys))}")

    if payload['group_mode'] != group_mode:
        raise ValueError(
            f"split JSON group_mode={payload['group_mode']!r} does not match requested group_mode={group_mode!r}"
        )

    expected_dataset_path = _normalize_dataset_path(dataset_path)
    payload_dataset_path = _normalize_dataset_path(payload.get('dataset_path')) if payload.get('dataset_path') else None
    if payload_dataset_path and expected_dataset_path and payload_dataset_path != expected_dataset_path:
        raise ValueError(
            f"split JSON dataset_path={payload_dataset_path!r} does not match requested dataset={expected_dataset_path!r}"
        )

    if payload.get('resolution_tag') != resolution_tag:
        raise ValueError(
            f"split JSON resolution_tag={payload.get('resolution_tag')!r} does not match requested "
            f"resolution_tag={resolution_tag!r}"
        )

    split_keys: dict[str, list[str]] = {}
    for split_name, json_key in SPLIT_GROUP_KEYS.items():
        value = payload[json_key]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"split JSON field {json_key!r} must be a list of group id strings")
        split_keys[split_name] = list(value)

    assigned = split_keys['train'] + split_keys['val'] + split_keys['test']
    duplicate_ids = sorted({group_id for group_id in assigned if assigned.count(group_id) > 1})
    if duplicate_ids:
        raise ValueError(f"split JSON assigns group(s) to multiple splits: {duplicate_ids}")

    existing = set(groups)
    requested = set(assigned)
    missing_groups = sorted(requested - existing)
    if missing_groups:
        raise ValueError(f"split JSON references group(s) not present in filtered dataset: {missing_groups}")

    unassigned_groups = sorted(existing - requested)
    if unassigned_groups:
        raise ValueError(f"split JSON does not assign filtered dataset group(s): {unassigned_groups}")

    return _split_info(
        split_keys['train'],
        split_keys['val'],
        split_keys['test'],
        int(payload['seed']),
        payload['group_mode'],
        payload.get('dataset_path'),
        payload.get('resolution_tag'),
    )


def split_dataset(
    dataset: list[Data],
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
    group_mode: str = 'legacy',
    filename_config: FilenameParseConfig | None = None,
    split_json_in: str | Path | None = None,
    split_json_out: str | Path | None = None,
    dataset_path: str | Path | None = None,
    resolution_tag: str | None = None,
) -> tuple[list[Data], list[Data], list[Data], dict]:
    """Grouped by base mesh to prevent augmentation leakage.

    The default legacy grouping only strips `_augN`. Use group_mode='family'
    to also group common resolution variants of the same mesh.

    Returns (train, val, test, split_info) where split_info maps
    split name -> list of base mesh names.
    """
    import random

    groups = _group_dataset(dataset, group_mode, filename_config)
    rng = random.Random(seed)

    if split_json_in:
        payload = load_split_json_metadata(split_json_in)
        split_info = _validate_split_payload(payload, groups, group_mode, dataset_path, resolution_tag)
        train_keys = split_info['train']
        val_keys = split_info['val']
        test_keys = split_info['test']
    else:
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        n = len(group_keys)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test_keys = group_keys[:n_test]
        val_keys = group_keys[n_test:n_test + n_val]
        train_keys = group_keys[n_test + n_val:]

        split_info = _split_info(
            train_keys,
            val_keys,
            test_keys,
            seed,
            group_mode,
            dataset_path,
            resolution_tag,
        )

    train = [d for k in train_keys for d in groups[k]]
    val = [d for k in val_keys for d in groups[k]]
    test = [d for k in test_keys for d in groups[k]]

    if split_json_out:
        save_split_json(split_json_out, split_info)

    return train, val, test, split_info


def load_dual_dataset(path: str | Path) -> list[Data]:
    from preprocessing.build_dual_graph import build_dual_graph_data
    original = load_dataset(path)
    return [build_dual_graph_data(d) for d in original]


def compute_pos_weight(dataset: list[Data], max_weight: float = 100.0) -> torch.Tensor:
    total_seam = sum(d.y.sum().item() for d in dataset)
    total_nonseam = sum((d.y == 0).sum().item() for d in dataset)
    weight = total_nonseam / max(total_seam, 1)
    if weight > max_weight:
        print(f"compute_pos_weight: clipping weight {weight:.4f} -> {max_weight:.4f}")
        weight = max_weight
    return torch.tensor([weight], dtype=torch.float32)
