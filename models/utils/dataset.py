import json
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.utils.filename_parsing import FilenameParseConfig, parse_mesh_name


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


def _group_name(d: Data, filename_config: FilenameParseConfig | None = None) -> str:
    path_or_name = getattr(d, 'file_path', '')
    if not path_or_name:
        return str(id(d))
    return parse_mesh_name(path_or_name, filename_config).family_id


def _group_dataset(
    dataset: list[Data],
    filename_config: FilenameParseConfig | None = None,
) -> dict[str, list[Data]]:
    groups: dict[str, list[Data]] = {}
    for d in dataset:
        groups.setdefault(_group_name(d, filename_config), []).append(d)
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
    dataset_path: str | Path | None,
    resolution_tag: str | None,
) -> dict:
    return {
        'train': sorted(train_keys),
        'val': sorted(val_keys),
        'test': sorted(test_keys),
        'seed': seed,
        'dataset_path': _normalize_dataset_path(dataset_path),
        'resolution_tag': resolution_tag,
    }


def _split_json_payload(split_info: dict) -> dict:
    return {
        'train_group_ids': split_info['train'],
        'val_group_ids': split_info['val'],
        'test_group_ids': split_info['test'],
        'seed': split_info.get('seed'),
        'dataset_path': split_info.get('dataset_path'),
        'resolution_tag': split_info.get('resolution_tag'),
    }


def _graph_weight(d: Data) -> int:
    for attr in ('edge_features', 'x', 'y'):
        value = getattr(d, attr, None)
        shape = getattr(value, 'shape', None)
        if shape and len(shape) > 0:
            return max(1, int(shape[0]))
    return 1


def _family_weight(graphs: list[Data]) -> int:
    return sum(_graph_weight(d) for d in graphs)


def _choose_forced_split(
    splits: list[str],
    targets: dict[str, float],
    assigned_weights: dict[str, int],
) -> str:
    return max(splits, key=lambda split: (targets[split] - assigned_weights[split], split == 'test'))


def _choose_improving_split(
    weight: int,
    splits: list[str],
    targets: dict[str, float],
    assigned_weights: dict[str, int],
) -> str | None:
    candidates = []
    for index, split in enumerate(splits):
        current = assigned_weights[split]
        target = targets[split]
        before = abs(target - current)
        after = abs(target - (current + weight))
        if after <= before:
            candidates.append((target - current, -after, -index, split))
    if not candidates:
        return None
    return max(candidates)[3]


def _weighted_split_group_keys(
    groups: dict[str, list[Data]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    import random

    group_keys = list(groups)
    if not group_keys:
        return [], [], []
    if len(group_keys) == 1:
        return group_keys, [], []

    rng = random.Random(seed)
    rng.shuffle(group_keys)
    weighted_groups = [
        (group_id, _family_weight(groups[group_id]), index)
        for index, group_id in enumerate(group_keys)
    ]
    weighted_groups.sort(key=lambda item: (-item[1], item[2]))

    train_keys: list[str] = []
    val_keys: list[str] = []
    test_keys: list[str] = []
    split_keys = {'train': train_keys, 'val': val_keys, 'test': test_keys}
    assigned_weights = {'train': 0, 'val': 0, 'test': 0}

    if len(weighted_groups) == 2:
        heldout_split = 'test' if test_ratio > 0 or val_ratio <= 0 else 'val'
        train_id, train_weight, _ = weighted_groups[0]
        heldout_id, heldout_weight, _ = weighted_groups[1]
        split_keys['train'].append(train_id)
        split_keys[heldout_split].append(heldout_id)
        assigned_weights['train'] += train_weight
        assigned_weights[heldout_split] += heldout_weight
        return train_keys, val_keys, test_keys

    total_weight = sum(weight for _, weight, _ in weighted_groups)
    active_splits = []
    if test_ratio > 0:
        active_splits.append('test')
    if val_ratio > 0:
        active_splits.append('val')
    targets = {
        'test': total_weight * max(0.0, test_ratio),
        'val': total_weight * max(0.0, val_ratio),
    }

    for index, (group_id, weight, _) in enumerate(weighted_groups):
        remaining_after = len(weighted_groups) - index - 1
        empty_required = [split for split in active_splits if not split_keys[split]]

        if not train_keys and remaining_after == len(empty_required):
            split = 'train'
        elif empty_required and remaining_after < len(empty_required):
            split = _choose_forced_split(empty_required, targets, assigned_weights)
        else:
            split = _choose_improving_split(weight, active_splits, targets, assigned_weights)
            if split is None:
                split = 'train'

        split_keys[split].append(group_id)
        assigned_weights[split] += weight

    return train_keys, val_keys, test_keys


def _validate_no_split_leakage(
    split_keys: dict[str, list[str]],
    groups: dict[str, list[Data]],
    filename_config: FilenameParseConfig | None = None,
) -> None:
    assigned = split_keys['train'] + split_keys['val'] + split_keys['test']
    duplicate_ids = sorted({group_id for group_id in assigned if assigned.count(group_id) > 1})
    if duplicate_ids:
        raise ValueError(f"split assigns group(s) to multiple splits: {duplicate_ids}")

    split_sets = {split: set(keys) for split, keys in split_keys.items()}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = sorted(split_sets[left] & split_sets[right])
        if overlap:
            raise ValueError(f"split group overlap between {left} and {right}: {overlap}")

    existing = set(groups)
    requested = set(assigned)
    missing_groups = sorted(requested - existing)
    if missing_groups:
        raise ValueError(f"split references group(s) not present in filtered dataset: {missing_groups}")

    unassigned_groups = sorted(existing - requested)
    if unassigned_groups:
        raise ValueError(f"split does not assign filtered dataset group(s): {unassigned_groups}")

    family_splits = {'train': set(), 'val': set(), 'test': set()}
    for split, group_ids in split_keys.items():
        for group_id in group_ids:
            for graph in groups[group_id]:
                path_or_name = getattr(graph, 'file_path', '')
                family_id = parse_mesh_name(path_or_name, filename_config).family_id if path_or_name else group_id
                family_splits[split].add(family_id)

    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = sorted(family_splits[left] & family_splits[right])
        if overlap:
            raise ValueError(f"family overlap between {left} and {right}: {overlap}")


def save_split_json(path: str | Path, split_info: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_split_json_payload(split_info), f, indent=2)


def _validate_split_payload(
    payload: dict,
    groups: dict[str, list[Data]],
    dataset_path: str | Path | None,
    resolution_tag: str | None,
) -> dict:
    missing_keys = [key for key in SPLIT_GROUP_KEYS.values() if key not in payload]
    missing_keys.extend(key for key in ('seed', 'dataset_path', 'resolution_tag') if key not in payload)
    if missing_keys:
        raise ValueError(f"split JSON missing required field(s): {', '.join(sorted(missing_keys))}")

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
        payload.get('dataset_path'),
        payload.get('resolution_tag'),
    )


def split_dataset(
    dataset: list[Data],
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
    filename_config: FilenameParseConfig | None = None,
    split_json_in: str | Path | None = None,
    split_json_out: str | Path | None = None,
    dataset_path: str | Path | None = None,
    resolution_tag: str | None = None,
) -> tuple[list[Data], list[Data], list[Data], dict]:
    """Grouped by mesh family to prevent augmentation and resolution leakage."""
    groups = _group_dataset(dataset, filename_config)

    if split_json_in:
        payload = load_split_json_metadata(split_json_in)
        split_info = _validate_split_payload(payload, groups, dataset_path, resolution_tag)
        train_keys = split_info['train']
        val_keys = split_info['val']
        test_keys = split_info['test']
    else:
        train_keys, val_keys, test_keys = _weighted_split_group_keys(groups, val_ratio, test_ratio, seed)

        split_info = _split_info(
            train_keys,
            val_keys,
            test_keys,
            seed,
            dataset_path,
            resolution_tag,
        )

    _validate_no_split_leakage(
        {'train': train_keys, 'val': val_keys, 'test': test_keys},
        groups,
        filename_config,
    )

    train = [d for k in train_keys for d in groups[k]]
    val = [d for k in val_keys for d in groups[k]]
    test = [d for k in test_keys for d in groups[k]]

    if split_json_out:
        save_split_json(split_json_out, split_info)

    return train, val, test, split_info


def load_dual_dataset(path: str | Path) -> list[Data]:
    from preprocessing.build_gnn_dataset import build_dual_data
    original = load_dataset(path)
    return [build_dual_data(d) for d in original]


def compute_pos_weight(dataset: list[Data], max_weight: float = 100.0) -> torch.Tensor:
    total_seam = sum(d.y.sum().item() for d in dataset)
    total_nonseam = sum((d.y == 0).sum().item() for d in dataset)
    weight = total_nonseam / max(total_seam, 1)
    if weight > max_weight:
        print(f"compute_pos_weight: clipping weight {weight:.4f} -> {max_weight:.4f}")
        weight = max_weight
    return torch.tensor([weight], dtype=torch.float32)
