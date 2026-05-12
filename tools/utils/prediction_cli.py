from __future__ import annotations

import argparse

from tools.utils.prediction_common import normalize_cli_model_type, normalize_feature_bundle_arg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict UV seam edges for a raw OBJ mesh.')
    parser.add_argument('--mesh-path', required=True)
    parser.add_argument('--model-weights', required=True)
    parser.add_argument(
        '--feature-bundle',
        default='auto',
        help='feature bundle: auto, paper14, or custom',
    )
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument(
        '--model-type',
        default='auto',
        help='model family: auto, graphsage, gatv2, or sparsemeshcnn',
    )
    parser.add_argument('--config-json', default=None)
    parser.add_argument('--enable-ao', action='store_true')
    parser.add_argument('--enable-dihedral', action='store_true')
    parser.add_argument('--enable-symmetry', action='store_true')
    parser.add_argument('--enable-density', action='store_true')
    parser.add_argument('--enable-thickness-sdf', action='store_true')
    parser.add_argument('--endpoint-seed', type=int, default=42)
    parser.add_argument('--write-all-edges', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--postprocess', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--postprocess-tau-low', type=float, default=0.30,
        help='Candidate-set threshold for skeletonization (Stage A).'
    )
    parser.add_argument(
        '--postprocess-d-max', type=int, default=3,
        help='Thickness-preservation distance for skeletonization (Stage A).'
    )
    parser.add_argument(
        '--postprocess-r-bridge', type=int, default=6,
        help='Default mesh-edge bridge radius for endpoint bridging (Stage B).'
    )
    parser.add_argument(
        '--postprocess-l-min', type=int, default=4,
        help='Minimum branch length for spur pruning (Stage C).'
    )
    parser.add_argument(
        '--postprocess-anchor-boundary',
        action=argparse.BooleanOptionalAction, default=True,
        help='Whether to use mesh-boundary vertices as structural anchors.'
    )
    args = parser.parse_args(argv)
    args.feature_bundle = normalize_feature_bundle_arg(args.feature_bundle)
    args.model_type = normalize_cli_model_type(args.model_type)
    return args
