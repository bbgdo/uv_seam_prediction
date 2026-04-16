from models.gatv2.model import DualGATv2


MODEL_CLASS = DualGATv2
DISPLAY_NAME = 'GATv2'
DEFAULT_CONFIG_OVERRIDES = {
    'hidden_size': 32,
    'lr': 5e-4,
    'heads': 4,
    'num_layers': 3,
    'dropout': 0.3,
}
