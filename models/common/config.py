from dataclasses import dataclass, replace
from typing import Any


DEFAULT_THRESHOLD_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


@dataclass(frozen=True)
class BaselineConfig:
    model_name: str = 'graphsage'
    hidden_size: int = 128
    num_layers: int = 3
    lr: float = 1e-3
    pos_weight: float | None = None
    focal_gamma: float = 2.0
    epochs: int = 100
    patience: int = 15
    threshold_values: tuple[float, ...] = DEFAULT_THRESHOLD_VALUES
    threshold_metric: str = 'f1'
    threshold_default: float = 0.5
    in_dim: int = 18
    dropout: float = 0.3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    heads: int = 8
    aggr: str = 'lstm'
    skip_connections: str = 'hidden'


def baseline_config(model_name: str, overrides: dict[str, Any] | None = None) -> BaselineConfig:
    values = {'model_name': model_name}
    if overrides:
        values.update(overrides)
    return replace(BaselineConfig(), **values)


def replace_config(config: BaselineConfig, **overrides: Any) -> BaselineConfig:
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **clean_overrides)
