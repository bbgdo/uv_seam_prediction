from dataclasses import dataclass, replace
from typing import Any


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
    in_dim: int = 18
    dropout: float = 0.3
    weight_decay: float = 1e-4
    heads: int = 8
    skip_connections: str = 'hidden'
    aggr: str = 'lstm'


def replace_config(config: GNNTrainConfig, **overrides: Any) -> GNNTrainConfig:
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **clean_overrides)
