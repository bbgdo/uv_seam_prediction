import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

# allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_dataset(path: str | Path) -> list[Data]:
    """Load a dataset saved by obj_to_dataset_graph.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    dataset = torch.load(path, weights_only=False)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"expected a non-empty list of Data objects, got: {type(dataset)}")

    return dataset


def split_dataset(
    dataset: list[Data],
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Train/val/test split grouped by base mesh to prevent augmentation leakage.

    Augmented copies (e.g. mesh_A_aug1.obj, mesh_A_aug2.obj) are always kept
    in the same split as their base mesh (mesh_A.obj).
    """
    import random
    import re

    def _base_name(d: Data) -> str:
        name = Path(getattr(d, 'file_path', '')).stem
        return re.sub(r'_aug\d+$', '', name) if name else str(id(d))

    # group graphs by base mesh name
    groups: dict[str, list[Data]] = {}
    for d in dataset:
        groups.setdefault(_base_name(d), []).append(d)

    # shuffle and split at the group level
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

    return train, val, test


def load_dual_dataset(path: str | Path) -> list[Data]:
    """Load original dataset and convert each graph to dual representation."""
    from preprocessing.build_dual_graph import build_dual_graph_data
    original = load_dataset(path)
    return [build_dual_graph_data(d) for d in original]


def compute_pos_weight(dataset: list[Data]) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight from train set seam/non-seam ratio."""
    total_seam = sum(d.y.sum().item() for d in dataset)
    total_nonseam = sum((d.y == 0).sum().item() for d in dataset)
    weight = total_nonseam / max(total_seam, 1)
    return torch.tensor([weight], dtype=torch.float32)
