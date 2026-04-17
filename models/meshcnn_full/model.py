from __future__ import annotations

import torch

from models.meshcnn_full.sparse_model import SparseMeshUNetSegmenter


class MeshCNNSegmenter(SparseMeshUNetSegmenter):
    """Drop-in sparse replacement for the previous MeshCNN-full segmenter."""


def build_model_from_checkpoint_payload(payload: dict, device: torch.device | str) -> MeshCNNSegmenter:
    config = dict(payload.get('model_config', {}))
    if 'in_channels' not in config:
        metadata = payload.get('feature_metadata', {})
        feature_names = metadata.get('feature_names') or []
        if not feature_names:
            raise ValueError('checkpoint is missing in_channels and feature_names metadata')
        config['in_channels'] = len(feature_names)
    model = MeshCNNSegmenter(**config).to(device)
    state = payload.get('model_state', payload)
    model.load_state_dict(state)
    model.eval()
    return model
