from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.meshcnn_full.model import build_model_from_checkpoint_payload
from models.utils.seam_topology import apply_topology_pipeline, build_seam_graph_view
from preprocessing.build_meshcnn_dataset import build_meshcnn_sample
from preprocessing.feature_registry import resolve_feature_selection
from preprocessing.obj_parser import parse_obj
from preprocessing.topology import WeldConfig, build_topology


def _feature_selection_from_metadata(metadata: dict[str, Any]):
    flags = dict(metadata.get('feature_flags') or {})
    group = metadata.get('feature_group') or metadata.get('feature_preset') or 'paper14'
    selection = resolve_feature_selection(
        group,
        enable_ao=bool(flags.get('ao', False)),
        enable_dihedral=bool(flags.get('signed_dihedral', False)),
        enable_symmetry=bool(flags.get('symmetry', False)),
        enable_density=bool(flags.get('density', False)),
        enable_thickness_sdf=bool(flags.get('thickness_sdf', False)),
    )
    expected_names = list(metadata.get('feature_names') or [])
    if expected_names and list(selection.feature_names) != expected_names:
        raise ValueError(
            'checkpoint feature_names do not match registry resolution; '
            'rebuild with matching feature metadata before inference'
        )
    return selection


@torch.no_grad()
def predict_obj(
    obj_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = dict(payload.get('feature_metadata') or {})
    selection = _feature_selection_from_metadata(metadata)
    endpoint_order = metadata.get('endpoint_order', 'auto')
    sample = build_meshcnn_sample(obj_path, selection, endpoint_order=endpoint_order)

    model = build_model_from_checkpoint_payload(payload, device)
    logits = model(sample)
    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    unique_edges = sample.unique_edges.detach().cpu().numpy().astype(np.int64)
    topology = build_topology(parse_obj(obj_path), WeldConfig.exact())
    return probs, unique_edges, topology


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run SparseMeshCNN UV seam inference.')
    parser.add_argument('--obj', required=True, help='Input OBJ path')
    parser.add_argument('--checkpoint', required=True, help='Path to best_model.pth')
    parser.add_argument('--output-probs', default=None, help='Optional .npz path for probabilities and edge indices')
    parser.add_argument('--output-seams', default=None, help='Optional txt path for thresholded seam edge indices')
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)

    probs, unique_edges, topology = predict_obj(args.obj, args.checkpoint, args.device)
    print(f'edges: {len(unique_edges)}')
    print(f'probabilities: min {probs.min():.4f}, mean {probs.mean():.4f}, max {probs.max():.4f}')

    if args.output_probs:
        out_path = Path(args.output_probs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, probs=probs, unique_edges=unique_edges)
        print(f'saved probabilities -> {out_path.resolve()}')

    if args.output_seams:
        view = build_seam_graph_view(topology, unique_edges)
        pipeline_result = apply_topology_pipeline(view, probs, topology=topology)
        mask = pipeline_result.final_edge_mask
        seam_indices = np.flatnonzero(mask)
        out_path = Path(args.output_seams)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('\n'.join(str(int(idx)) for idx in seam_indices), encoding='utf-8')
        print(f'saved seams -> {out_path.resolve()} ({len(seam_indices)} edges)')


if __name__ == '__main__':
    main()
