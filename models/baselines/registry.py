from dataclasses import dataclass

from torch.nn import Module

from models.baselines import gatv2, graphsage


@dataclass(frozen=True)
class BaselineDefinition:
    model_class: type[Module]
    display_name: str
    default_config_overrides: dict


BASELINES = {
    'graphsage': BaselineDefinition(
        model_class=graphsage.MODEL_CLASS,
        display_name=graphsage.DISPLAY_NAME,
        default_config_overrides=graphsage.DEFAULT_CONFIG_OVERRIDES,
    ),
    'gatv2': BaselineDefinition(
        model_class=gatv2.MODEL_CLASS,
        display_name=gatv2.DISPLAY_NAME,
        default_config_overrides=gatv2.DEFAULT_CONFIG_OVERRIDES,
    ),
}

SUPPORTED_BASELINES = tuple(BASELINES)


def get_baseline(model_name: str) -> BaselineDefinition:
    try:
        return BASELINES[model_name]
    except KeyError as exc:
        raise ValueError(f"unsupported baseline model {model_name!r}; choose one of {SUPPORTED_BASELINES}") from exc
