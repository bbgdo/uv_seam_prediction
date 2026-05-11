from dataclasses import dataclass, replace
from typing import Any


DEFAULT_THRESHOLD_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


@dataclass(frozen=True)
class GNNTrainConfig:
    model_name: str = 'graphsage'
    hidden_size: int = 128
    num_layers: int = 3
    lr: float = 3e-4
    pos_weight: float | None = None
    focal_gamma: float = 2.0
    epochs: int = 100
    patience: int = 15
    threshold_values: tuple[float, ...] = DEFAULT_THRESHOLD_VALUES
    in_dim: int = 18
    dropout: float = 0.3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    heads: int = 8
    aggr: str = 'lstm'
    skip_connections: str = 'hidden'


def gnn_train_config(model_name: str, overrides: dict[str, Any] | None = None) -> GNNTrainConfig:
    values = {'model_name': model_name}
    if overrides:
        values.update(overrides)
    return replace(GNNTrainConfig(), **values)


def replace_config(config: GNNTrainConfig, **overrides: Any) -> GNNTrainConfig:
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **clean_overrides)
