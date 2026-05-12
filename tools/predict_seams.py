from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from tools._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from models.utils.seam_topology import apply_topology_pipeline, build_seam_graph_view, topology_pipeline_result_to_json_dict  # noqa: E402
from preprocessing.compute_features import compute_edge_features_for_selection  # noqa: E402
from preprocessing.feature_registry import resolve_feature_selection  # noqa: E402
from preprocessing.obj_parser import parse_obj  # noqa: E402
from preprocessing.topology import WeldConfig, build_topology  # noqa: E402
from tools.utils.prediction_cli import parse_args  # noqa: E402
from tools.utils.prediction_common import (  # noqa: E402
    FEATURE_BUNDLES,
    MODEL_TYPES,
    PredictionError,
    load_json_object,
    normalize_cli_model_type,
    require_file,
    resolve_threshold,
)
from tools.utils.prediction_features import (  # noqa: E402
    infer_feature_bundle,
    resolve_feature_bundle,
    validate_feature_metadata,
)
from tools.utils.prediction_mesh import (  # noqa: E402
    assert_canonical_edge_order,
    build_dual_data,
    build_feature_mesh_from_canonical_topology,
    build_mesh_diagnostics,
    build_meshcnn_inference_sample,
)
from tools.utils.prediction_models import (  # noqa: E402
    build_prediction_model,
    extract_state_dict,
    load_weights_payload,
    normalize_probabilities,
    resolve_device,
    resolve_model_kwargs,
    resolve_model_type,
)
from tools.utils.prediction_output import build_output_payload, write_error_payload, write_json_payload  # noqa: E402


_normalize_cli_model_type = normalize_cli_model_type
__all__ = [
    'FEATURE_BUNDLES',
    'MODEL_TYPES',
    'PredictionError',
    '_normalize_cli_model_type',
    'apply_topology_pipeline',
    'assert_canonical_edge_order',
    'build_dual_data',
    'build_feature_mesh_from_canonical_topology',
    'build_mesh_diagnostics',
    'build_meshcnn_inference_sample',
    'build_output_payload',
    'build_prediction_model',
    'build_seam_graph_view',
    'compute_edge_features_for_selection',
    'extract_state_dict',
    'infer_feature_bundle',
    'load_json_object',
    'load_weights_payload',
    'normalize_probabilities',
    'parse_args',
    'postprocess_kwargs_from_args',
    'require_file',
    'resolve_device',
    'resolve_feature_bundle',
    'resolve_feature_selection',
    'resolve_model_kwargs',
    'resolve_model_type',
    'resolve_threshold',
    'run_prediction',
    'topology_pipeline_result_to_json_dict',
    'validate_feature_metadata',
    'write_error_payload',
    'write_json_payload',
]


def postprocess_kwargs_from_args(args) -> dict[str, Any]:
    return {
        'tau_low': float(args.postprocess_tau_low),
        'd_max': int(args.postprocess_d_max),
        'r_bridge': int(args.postprocess_r_bridge),
        'l_min': int(args.postprocess_l_min),
        'anchor_boundary': bool(args.postprocess_anchor_boundary),
    }


def run_prediction(args) -> dict[str, Any]:
    mesh_path = Path(args.mesh_path)
    weights_path = Path(args.model_weights)
    output_json = Path(args.output_json)
    config_path = Path(args.config_json) if args.config_json else weights_path.with_name('config.json')

    require_file(mesh_path, 'input OBJ')
    require_file(weights_path, 'model weights')
    require_file(config_path, 'config JSON')

    config = load_json_object(config_path, 'config JSON')
    model_type = resolve_model_type(args.model_type, config, weights_path)
    threshold = resolve_threshold(args.threshold)
    selection, endpoint_order, resolved_feature_bundle = resolve_feature_bundle(args, config)
    device = resolve_device(args.device)
    model_kwargs = resolve_model_kwargs(model_type, config)
    validate_feature_metadata(config, selection, model_kwargs)

    obj_mesh = parse_obj(mesh_path)
    topology = build_topology(obj_mesh, WeldConfig.exact())
    feature_mesh = build_feature_mesh_from_canonical_topology(topology)

    with contextlib.redirect_stdout(io.StringIO()):
        edge_features, unique_edges, _ = compute_edge_features_for_selection(
            feature_mesh,
            selection,
            endpoint_order=endpoint_order,
            rng_seed=args.endpoint_seed,
        )
    assert_canonical_edge_order(unique_edges, topology.canonical_edges, mesh_path)
    model = build_prediction_model(model_type, model_kwargs)
    weights_payload = load_weights_payload(weights_path, device)
    state_dict = extract_state_dict(weights_payload)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise PredictionError(f'model state dict did not load cleanly: {exc}', 'InvalidWeights') from exc
    model.to(device)
    model.eval()

    with torch.no_grad():
        if model_type == 'sparsemeshcnn':
            sample = build_meshcnn_inference_sample(
                mesh_path=mesh_path,
                feature_mesh=feature_mesh,
                unique_edges=unique_edges,
                edge_features=edge_features,
                selection=selection,
                endpoint_order=endpoint_order,
                topology=topology,
            )
            logits = model(sample)
        else:
            dual_data = build_dual_data(edge_features, unique_edges)
            logits = model(dual_data.x.to(device), dual_data.edge_index.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    probabilities = normalize_probabilities(probs, len(unique_edges))
    diagnostics = build_mesh_diagnostics(
        feature_mesh,
        edge_features,
        probabilities=probabilities,
        threshold=threshold,
        topology=topology,
        unique_edges=unique_edges,
    )
    pipeline_telemetry: dict[str, Any] | None = None
    if not bool(getattr(args, 'postprocess', True)):
        seam_mask = probabilities >= threshold
    else:
        try:
            view = build_seam_graph_view(topology, unique_edges)
            pp_kwargs = postprocess_kwargs_from_args(args)
            pipeline_result = apply_topology_pipeline(
                view=view,
                probabilities=probabilities,
                topology=topology,
                **pp_kwargs,
            )
        except Exception as exc:
            raise PredictionError(
                f'postprocess pipeline failed: {exc}',
                'PostprocessFailed',
            ) from exc
        seam_mask = pipeline_result.final_edge_mask
        pipeline_telemetry = topology_pipeline_result_to_json_dict(pipeline_result)

    if pipeline_telemetry is not None:
        diagnostics['postprocess'] = pipeline_telemetry

    return build_output_payload(
        mesh_path=mesh_path,
        output_json=output_json,
        weights_path=weights_path,
        config_path=config_path,
        model_type=model_type,
        feature_bundle=resolved_feature_bundle,
        selection=selection,
        threshold=threshold,
        device=device,
        topology=topology,
        unique_edges=unique_edges,
        probabilities=probabilities,
        seam_mask=seam_mask,
        write_all_edges=args.write_all_edges,
        diagnostics=diagnostics,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = Path(args.output_json)
    try:
        payload = run_prediction(args)
        write_json_payload(output_json, payload)
        bbox = payload.get('diagnostics', {}).get('mesh_bbox', {})
        bbox_diag = bbox.get('diagonal')
        bbox_size = bbox.get('size')
        if bbox_diag is not None and bbox_size is not None:
            print(f'mesh bbox size {bbox_size}, diagonal {bbox_diag:.9g}')
        print(
            f"predicted {payload['stats']['predicted_seam_count']} seam edges "
            f"out of {payload['topology']['edge_count']} -> {output_json.resolve()}"
        )
        return 0
    except Exception as exc:
        error_type = getattr(exc, 'error_type', exc.__class__.__name__)
        message = str(exc)
        try:
            write_error_payload(output_json, error_type, message)
        except Exception:
            pass
        print(f'{error_type}: {message}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
