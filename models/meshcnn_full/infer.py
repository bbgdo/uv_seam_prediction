from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.meshcnn_full.model import build_model_from_checkpoint_payload
from models.utils.postprocess import postprocess_seams
from preprocessing.build_meshcnn_dataset_v2 import build_meshcnn_sample
from preprocessing.feature_registry import resolve_feature_selection


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


def _edge_to_faces_dict(unique_edges: np.ndarray, edge_to_faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for edge, faces in zip(unique_edges, edge_to_faces):
        result[(int(edge[0]), int(edge[1]))] = [int(face) for face in faces if int(face) >= 0]
    return result


@torch.no_grad()
def predict_obj(
    obj_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], list[int]]]:
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
    edge_to_faces = _edge_to_faces_dict(unique_edges, sample.edge_to_faces.detach().cpu().numpy())
    return probs, unique_edges, edge_to_faces


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run isolated MeshCNN-full UV seam inference.')
    parser.add_argument('--obj', required=True, help='Input OBJ path')
    parser.add_argument('--checkpoint', required=True, help='Path to best_model.pth')
    parser.add_argument('--output-probs', default=None, help='Optional .npz path for probabilities and edge indices')
    parser.add_argument('--output-seams', default=None, help='Optional txt path for thresholded seam edge indices')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-component-size', type=int, default=3)
    parser.add_argument('--max-gap', type=int, default=3)
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)

    probs, unique_edges, edge_to_faces = predict_obj(args.obj, args.checkpoint, args.device)
    print(f'edges: {len(unique_edges)}')
    print(f'probabilities: min {probs.min():.4f}, mean {probs.mean():.4f}, max {probs.max():.4f}')

    if args.output_probs:
        out_path = Path(args.output_probs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, probs=probs, unique_edges=unique_edges)
        print(f'saved probabilities -> {out_path.resolve()}')

    if args.output_seams:
        mask = postprocess_seams(
            probs,
            unique_edges,
            edge_to_faces=edge_to_faces,
            threshold=args.threshold,
            min_component_size=args.min_component_size,
            max_gap=args.max_gap,
        )
        seam_indices = np.flatnonzero(mask)
        out_path = Path(args.output_seams)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('\n'.join(str(int(idx)) for idx in seam_indices), encoding='utf-8')
        print(f'saved seams -> {out_path.resolve()} ({len(seam_indices)} edges)')


if __name__ == '__main__':
    main()
