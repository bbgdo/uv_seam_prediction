import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.utils.filename_parsing import FilenameParseConfig, legacy_base_name, parse_mesh_name


def load_dataset(path: str | Path) -> list[Data]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    dataset = torch.load(path, weights_only=False)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"expected a non-empty list of Data objects, got: {type(dataset)}")

    return dataset


def filter_dataset_by_resolution(dataset: list[Data], resolution_tag: str | None) -> list[Data]:
    if not resolution_tag:
        return dataset

    filtered = [
        d for d in dataset
        if parse_mesh_name(getattr(d, 'file_path', '')).resolution_tag == resolution_tag
    ]
    if not filtered:
        raise ValueError(f"no graphs matched resolution tag: {resolution_tag}")
    return filtered


def split_dataset(
    dataset: list[Data],
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
    group_mode: str = 'legacy',
    filename_config: FilenameParseConfig | None = None,
) -> tuple[list[Data], list[Data], list[Data], dict]:
    """Grouped by base mesh to prevent augmentation leakage.

    The default legacy grouping only strips `_augN`. Use group_mode='family'
    to also group common resolution variants of the same mesh.

    Returns (train, val, test, split_info) where split_info maps
    split name -> list of base mesh names.
    """
    import random

    if group_mode not in {'legacy', 'family'}:
        raise ValueError(f"group_mode must be 'legacy' or 'family', got: {group_mode}")

    def _group_name(d: Data) -> str:
        name = Path(getattr(d, 'file_path', '')).stem
        if not name:
            return str(id(d))
        if group_mode == 'family':
            return parse_mesh_name(name, filename_config).family_id
        return legacy_base_name(name)

    groups: dict[str, list[Data]] = {}
    for d in dataset:
        groups.setdefault(_group_name(d), []).append(d)

    rng = random.Random(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_keys = group_keys[:n_test]
    val_keys = group_keys[n_test:n_test + n_val]
    train_keys = group_keys[n_test + n_val:]

    train = [d for k in train_keys for d in groups[k]]
    val = [d for k in val_keys for d in groups[k]]
    test = [d for k in test_keys for d in groups[k]]

    split_info = {
        'train': sorted(train_keys),
        'val': sorted(val_keys),
        'test': sorted(test_keys),
    }

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
