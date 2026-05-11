from models.gatv2.model import DualGATv2


MODEL_CLASS = DualGATv2
DISPLAY_NAME = 'GATv2'

DEFAULT_CONFIG_OVERRIDES = {
    'hidden_size': 64,
    'heads': 4,
    'num_layers': 4,
    'dropout': 0.2,
}
