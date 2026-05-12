from dataclasses import dataclass

from torch.nn import Module

from models.common.gnn_config import GNNTrainConfig
from models.dual_graphsage.model import DualGraphSAGE
from models.gatv2.model import DualGATv2


@dataclass(frozen=True)
class GNNModelDefinition:
    model_class: type[Module]
    display_name: str
    train_config: GNNTrainConfig


GNN_MODEL_DEFINITIONS = {
    'graphsage': GNNModelDefinition(
        model_class=DualGraphSAGE,
        display_name='DualGraphSAGE',
        train_config=GNNTrainConfig(model_name='graphsage'),
    ),
    'gatv2': GNNModelDefinition(
        model_class=DualGATv2,
        display_name='GATv2',
        train_config=GNNTrainConfig(
            model_name='gatv2',
            hidden_size=64,
            heads=4,
            num_layers=4,
            dropout=0.2,
        ),
    ),
}

SUPPORTED_GNN_MODELS = tuple(GNN_MODEL_DEFINITIONS)


def get_gnn_model(model_name: str) -> GNNModelDefinition:
    try:
        return GNN_MODEL_DEFINITIONS[model_name]
    except KeyError as exc:
        raise ValueError(f"unsupported GNN model {model_name!r}; choose one of {SUPPORTED_GNN_MODELS}") from exc
