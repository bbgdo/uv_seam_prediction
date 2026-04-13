from models.dual_graphsage.model import DualGraphSAGE


MODEL_CLASS = DualGraphSAGE
DISPLAY_NAME = 'DualGraphSAGE'
DEFAULT_CONFIG_OVERRIDES = {
    'hidden_size': 128,
    'lr': 1e-3,
}
