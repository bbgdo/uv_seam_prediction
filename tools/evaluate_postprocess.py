from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.baselines.registry import get_baseline  # noqa: E402
from models.common.config import baseline_config, replace_config  # noqa: E402
from models.common.baseline_train import apply_runtime_feature_selection  # noqa: E402
from models.meshcnn_full.mesh import load_meshcnn_dataset  # noqa: E402
from models.utils.dataset import filter_dataset_by_resolution, load_dataset  # noqa: E402
from models.utils.filename_parsing import legacy_base_name, parse_mesh_name  # noqa: E402
from models.utils.metrics import binary_metrics_from_probs  # noqa: E402
from models.utils.postprocess import apply_seam_postprocessing_detailed  # noqa: E402
from preprocessing.feature_registry import resolve_feature_selection  # noqa: E402
from preprocessing.obj_parser import parse_obj, parse_obj_text  # noqa: E402
from preprocessing.topology import WeldConfig, build_topology  # noqa: E402
from tools import predict_seams as predict_bridge  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate seam post-processing on a checkpoint test split.')
    parser.add_argument('--model-weights', required=True)
    parser.add_argument('--config-json', default=None)
    parser.add_argument('--summary-json', default=None)
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--threshold', type=float, default=0.60)
    parser.add_argument('--max-gap-length', type=int, default=3)
    parser.add_argument('--min-island-size', type=int, default=3)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--limit-meshes', type=int, default=None)
    parser.add_argument('--mesh-path', type=str, default=None)
    parser.add_argument('--triangulate', action='store_true')
    parser.add_argument('--debug-export', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--output-json', default=None)
    return parser.parse_args(argv)


def _resolve_existing_path(raw_path: str | Path | None, *bases: Path) -> Path | None:
    if raw_path in (None, ''):
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    search_roots = list(bases) + [REPO_ROOT]
    for root in search_roots:
        path = (root / candidate).resolve(strict=False)
        if path.exists():
            return path
    return candidate.resolve(strict=False) if candidate.is_absolute() else (REPO_ROOT / candidate).resolve(strict=False)


def _group_name(file_path: str | Path, group_mode: str) -> str:
    if group_mode == 'family':
        return parse_mesh_name(file_path).family_id
    return legacy_base_name(file_path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')


def _resolve_feature_selection(
    config: dict[str, Any],
    summary: dict[str, Any],
    expected_dim: int,
):
    try:
        selection, _, _ = predict_bridge.infer_feature_bundle(config, summary)
        if selection.feature_count == expected_dim:
            return selection
    except Exception:
        pass

    candidates = (
        resolve_feature_selection('paper14'),
        resolve_feature_selection('extended18'),
        resolve_feature_selection('custom', enable_ao=True, enable_density=True),
        resolve_feature_selection('custom', enable_thickness_sdf=True),
    )
    for candidate in candidates:
        if candidate.feature_count == expected_dim:
            return candidate
    return None


def _load_dataset_for_run(
    *,
    model_type: str,
    dataset_path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
):
    if model_type == 'meshcnn_full':
        return load_meshcnn_dataset(dataset_path)

    dataset = load_dataset(dataset_path)
    expected_dim = int(_resolve_model_kwargs_for_evaluation(model_type, config)['in_dim'])
    selection = _resolve_feature_selection(config, summary, expected_dim)
    if selection is not None:
        try:
            dataset = apply_runtime_feature_selection(dataset, selection)
        except ValueError as exc:
            current_dim = int(dataset[0].x.shape[1])
            if 'missing feature_names metadata' not in str(exc) or current_dim != expected_dim:
                raise
    actual_dim = int(dataset[0].x.shape[1])
    if actual_dim != expected_dim:
        raise ValueError(
            f'dataset feature dim {actual_dim} does not match model input dim {expected_dim} for {model_type}'
        )
    return dataset


def _resolve_model_kwargs_for_evaluation(model_type: str, config: dict[str, Any]) -> dict[str, Any]:
    try:
        return predict_bridge.resolve_model_kwargs(model_type, config)
    except predict_bridge.PredictionError:
        if model_type == 'meshcnn_full':
            raise

    definition = get_baseline(model_type)
    runtime = replace_config(
        baseline_config(model_type, definition.default_config_overrides),
        hidden_size=config.get('hidden_dim', config.get('hidden_size')),
        num_layers=config.get('num_layers'),
        in_dim=config.get('in_dim'),
        dropout=config.get('dropout'),
        heads=config.get('heads'),
        aggr=config.get('aggr'),
        skip_connections=config.get('skip_connections'),
    )
    kwargs = {
        'in_dim': int(runtime.in_dim),
        'hidden_dim': int(runtime.hidden_size),
        'num_layers': int(runtime.num_layers),
        'dropout': float(runtime.dropout),
    }
    if model_type == 'graphsage':
        kwargs['aggr'] = str(runtime.aggr)
        kwargs['skip_connections'] = str(runtime.skip_connections)
    elif model_type == 'gatv2':
        kwargs['heads'] = int(runtime.heads)
    return kwargs


def _select_test_samples(
    dataset,
    split_info: dict[str, Any],
    limit_meshes: int | None,
):
    group_mode = str(split_info.get('group_mode') or 'legacy')
    test_groups = set(split_info.get('test') or ())
    if not test_groups:
        raise ValueError('config JSON is missing split.test entries')

    selected = [sample for sample in dataset if _group_name(getattr(sample, 'file_path', ''), group_mode) in test_groups]
    if not selected:
        raise ValueError(f'no dataset samples matched split.test groups for group_mode={group_mode}')
    if limit_meshes is not None:
        selected = selected[:limit_meshes]
    return selected


def _resolve_mesh_path(file_path: str | Path) -> Path:
    path = _resolve_existing_path(file_path, REPO_ROOT)
    if path is None or not path.exists():
        raise FileNotFoundError(f'mesh file not found: {file_path}')
    return path


def _sample_key(file_path: str | Path) -> str:
    try:
        return str(_resolve_mesh_path(file_path))
    except FileNotFoundError:
        return str(file_path)


def _resolve_original_dataset_path(dual_dataset_path: Path) -> Path | None:
    candidates = []
    name = dual_dataset_path.name
    if '_dual' in dual_dataset_path.stem:
        candidates.append(dual_dataset_path.with_name(name.replace('_dual', '', 1)))
    if name == 'dataset_dual.pt':
        candidates.append(dual_dataset_path.with_name('dataset.pt'))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_original_dataset_lookup(dataset_path: Path) -> dict[str, Any]:
    original_path = _resolve_original_dataset_path(dataset_path)
    if original_path is None:
        return {}
    original_dataset = load_dataset(original_path)
    return {
        _sample_key(getattr(sample, 'file_path', '')): sample
        for sample in original_dataset
    }


def _unique_edges_from_original_sample(original_sample: Any) -> np.ndarray:
    num_directed = int(original_sample.edge_index.shape[1])
    num_unique = num_directed // 2
    return original_sample.edge_index[:, :num_unique].detach().cpu().numpy().T.astype(np.int64)


def _topology_cache_key(path: Path) -> str:
    return str(path.resolve())


def _load_topology(mesh_path: Path, cache: dict[str, Any]):
    key = _topology_cache_key(mesh_path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    topology = build_topology(parse_obj(mesh_path), WeldConfig.exact())
    unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64)
    cache[key] = (topology, unique_edges)
    return cache[key]


def _triangulate_obj_mesh(mesh_path: Path):
    try:
        loaded = trimesh.load(mesh_path, force='mesh', process=False, maintain_order=True)
    except TypeError:
        loaded = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f'failed to load a single mesh from {mesh_path}')

    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f'invalid vertex array from triangulation for {mesh_path}')
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f'triangulation did not produce triangle faces for {mesh_path}')

    lines = ['# evaluate_postprocess triangulated OBJ']
    lines.extend(f'v {x:.9g} {y:.9g} {z:.9g}' for x, y, z in vertices)
    lines.extend(f'f {a + 1} {b + 1} {c + 1}' for a, b, c in faces)
    return parse_obj_text('\n'.join(lines) + '\n', file_path=str(mesh_path))


def _load_topology_for_standalone_mesh(mesh_path: Path, triangulate: bool):
    if triangulate:
        obj_mesh = _triangulate_obj_mesh(mesh_path)
        topology = build_topology(obj_mesh, WeldConfig.exact())
        unique_edges = np.asarray(topology.canonical_edges, dtype=np.int64)
        return topology, unique_edges
    return _load_topology(mesh_path, {})


def _binary_metrics_from_mask(mask: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    preds = np.asarray(mask, dtype=bool).reshape(-1)
    gt = np.asarray(labels, dtype=bool).reshape(-1)
    if preds.shape != gt.shape:
        raise ValueError(f'prediction mask shape {preds.shape} does not match labels shape {gt.shape}')

    tp = int(np.count_nonzero(preds & gt))
    fp = int(np.count_nonzero(preds & ~gt))
    fn = int(np.count_nonzero(~preds & gt))
    tn = int(np.count_nonzero(~preds & ~gt))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(gt), 1)
    fpr = fp / max(fp + tn, 1)
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def _metric_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(after[key]) - float(before[key])
        for key in ('f1', 'precision', 'recall', 'accuracy', 'fpr', 'tpr')
    }


def _infer_probabilities(model_type: str, model: torch.nn.Module, sample: Any, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        if model_type == 'meshcnn_full':
            logits = model(sample)
        else:
            logits = model(sample.x.to(device), sample.edge_index.to(device))
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64).reshape(-1)


def _load_model_state_for_evaluation(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    model_type: str,
) -> dict[str, Any]:
    try:
        model.load_state_dict(state_dict, strict=True)
        return {'strict': True, 'missing_keys': [], 'unexpected_keys': []}
    except RuntimeError as exc:
        if model_type != 'graphsage':
            raise
        load_result = model.load_state_dict(state_dict, strict=False)
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        if unexpected or any(not key.startswith('skips.') for key in missing):
            raise exc
        return {'strict': False, 'missing_keys': missing, 'unexpected_keys': unexpected}


def _build_single_mesh_inference_inputs(
    mesh_path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    model_type: str,
    model_kwargs: dict[str, Any],
    triangulate: bool,
) -> tuple[Any, np.ndarray, np.ndarray]:
    prediction_args = argparse.Namespace(
        feature_bundle='auto',
        enable_ao=False,
        enable_dihedral=False,
        enable_symmetry=False,
        enable_density=False,
        threshold=None,
        fail_if_threshold_missing=False,
        endpoint_seed=42,
    )
    selection, endpoint_order, _resolved_feature_bundle = predict_bridge.resolve_feature_bundle(
        prediction_args,
        config,
        summary,
    )
    predict_bridge.validate_feature_metadata(config, summary, selection, model_kwargs)

    topology, unique_edges = _load_topology_for_standalone_mesh(mesh_path, triangulate)
    feature_mesh = predict_bridge.build_feature_mesh_from_canonical_topology(topology)
    with contextlib.redirect_stdout(io.StringIO()):
        edge_features, feature_unique_edges, _ = predict_bridge.compute_edge_features_for_selection(
            feature_mesh,
            selection,
            endpoint_order=endpoint_order,
            rng_seed=prediction_args.endpoint_seed,
        )
    predict_bridge.assert_canonical_edge_order(feature_unique_edges, topology.canonical_edges, mesh_path)

    if model_type == 'meshcnn_full':
        model_input = predict_bridge.build_meshcnn_inference_sample(
            mesh_path=mesh_path,
            feature_mesh=feature_mesh,
            unique_edges=feature_unique_edges,
            edge_features=edge_features,
            selection=selection,
            endpoint_order=endpoint_order,
            topology=topology,
        )
    else:
        model_input = predict_bridge.build_dual_data(edge_features, feature_unique_edges)
    return topology, feature_unique_edges.astype(np.int64, copy=False), model_input


def _infer_single_mesh_probabilities(
    model_type: str,
    model: torch.nn.Module,
    model_input: Any,
    unique_edges: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        if model_type == 'meshcnn_full':
            logits = model(model_input)
        else:
            logits = model(model_input.x.to(device), model_input.edge_index.to(device))
    return predict_bridge.normalize_probabilities(
        torch.sigmoid(logits).detach().cpu().numpy(),
        len(unique_edges),
    )


def _evaluate_arbitrary_mesh(
    *,
    args: argparse.Namespace,
    mesh_path: Path,
    weights_path: Path,
    config_path: Path,
    summary_path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    model_type: str,
    device: torch.device,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    threshold = predict_bridge.resolve_threshold(args.threshold, summary, fail_if_missing=True)
    topology, unique_edges, model_input = _build_single_mesh_inference_inputs(
        mesh_path,
        config,
        summary,
        model_type,
        model_kwargs,
        bool(getattr(args, 'triangulate', False)),
    )

    model = predict_bridge.build_prediction_model(model_type, model_kwargs).to(device)
    state_dict = predict_bridge.extract_state_dict(predict_bridge.load_weights_payload(weights_path, device))
    state_dict_load = _load_model_state_for_evaluation(model, state_dict, model_type)
    model.eval()

    probabilities = _infer_single_mesh_probabilities(model_type, model, model_input, unique_edges, device)
    debug_export_dir = (REPO_ROOT / 'debug_output') if bool(getattr(args, 'debug_export', False)) else None
    result = apply_seam_postprocessing_detailed(
        topology=topology,
        unique_edges=unique_edges,
        probabilities=probabilities,
        threshold=threshold,
        max_gap_length=args.max_gap_length,
        min_island_size=args.min_island_size,
        debug_export_dir=debug_export_dir,
    )
    raw_mask = probabilities >= threshold
    meshes_with_changes = int(np.any(raw_mask != result.final_mask))

    per_mesh = [{
        'file_path': str(mesh_path),
        'edge_count': int(len(unique_edges)),
        'gt_seam_count': None,
        'raw_predicted_count': int(np.count_nonzero(raw_mask)),
        'post_predicted_count': int(np.count_nonzero(result.final_mask)),
        'raw_metrics': None,
        'post_metrics': None,
        'delta': None,
        'postprocess': {
            'skeleton_deleted_vertices': [int(idx) for idx in result.skeleton_deleted_vertices],
            'skeleton_edge_count': int(np.count_nonzero(result.skeleton_mask)),
            'steiner_added_edges': [int(idx) for idx in result.steiner_added_edges],
            'pruned_edge_indices': [int(idx) for idx in result.pruned_edge_indices],
            'steiner_edge_count': int(np.count_nonzero(result.steiner_mask)),
            'skeleton_terminal_vertex_count': int(result.skeleton_terminal_vertex_count),
            'steiner_terminal_group_count': int(result.steiner_terminal_group_count),
            'steiner_tree_count': int(result.steiner_tree_count),
            'pruned_component_count': int(result.pruned_component_count),
        },
    }]

    return {
        'status': 'completed',
        'checkpoint': {
            'model_weights': str(weights_path),
            'config_json': str(config_path),
            'summary_json': str(summary_path),
            'dataset_path': None,
        },
        'model': {
            'model_type': model_type,
            'model_kwargs': model_kwargs,
            'device': str(device),
            'state_dict_load': state_dict_load,
        },
        'split': {
            'group_mode': None,
            'seed': None,
            'resolution_tag': None,
            'test_group_ids': [],
            'evaluated_mesh_count': 1,
            'requested_mesh_path': str(mesh_path.resolve()),
        },
        'postprocess': {
            'threshold': float(threshold),
            'max_gap_length': int(args.max_gap_length),
            'min_island_size': int(args.min_island_size),
            'debug_export': bool(getattr(args, 'debug_export', False)),
            'debug_export_dir': None if debug_export_dir is None else str(debug_export_dir.resolve()),
            'pipeline': (
                'local support smoothing, hysteresis thresholding, tiny component pruning, '
                'low-confidence spur pruning, terminal-set bridging with confidence-gated acceptance, final cleanup'
            ),
            'bridge_cost_function': '1 + alpha_cost * (1 - seam_probability)',
            'standalone_mesh_mode': True,
            'triangulate': bool(getattr(args, 'triangulate', False)),
        },
        'global_metrics': {
            'pre_pp': None,
            'post_pp': None,
            'delta': None,
        },
        'aggregate_postprocess': {
            'meshes_with_changes': int(meshes_with_changes),
            'total_added_edges': int(len(result.steiner_added_edges)),
            'total_pruned_edges': int(len(result.pruned_edge_indices)),
            'total_bridges': int(result.steiner_tree_count),
            'total_steiner_trees': int(result.steiner_tree_count),
        },
        'per_mesh': per_mesh,
    }


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    weights_path = Path(args.model_weights).resolve()
    config_path = Path(args.config_json).resolve() if args.config_json else weights_path.with_name('config.json')
    summary_path = Path(args.summary_json).resolve() if args.summary_json else weights_path.with_name('summary.json')

    config = predict_bridge.load_json(config_path, 'config JSON')
    summary = predict_bridge.load_json(summary_path, 'summary JSON') if summary_path.exists() else {}
    model_type = predict_bridge.resolve_model_type(args.model_type if hasattr(args, 'model_type') else 'auto', config, weights_path)
    device = predict_bridge.resolve_device(args.device)
    model_kwargs = _resolve_model_kwargs_for_evaluation(model_type, config)

    if args.mesh_path not in (None, ''):
        mesh_path = _resolve_mesh_path(args.mesh_path)
        return _evaluate_arbitrary_mesh(
            args=args,
            mesh_path=mesh_path,
            weights_path=weights_path,
            config_path=config_path,
            summary_path=summary_path,
            config=config,
            summary=summary,
            model_type=model_type,
            device=device,
            model_kwargs=model_kwargs,
        )

    dataset_hint = args.dataset_path or config.get('dataset') or config.get('split', {}).get('dataset_path')
    if not dataset_hint:
        raise ValueError('dataset path is missing from config JSON; pass --dataset-path explicitly')
    dataset_path = _resolve_existing_path(dataset_hint, config_path.parent, weights_path.parent)
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(f'dataset path not found: {dataset_hint}')

    dataset = _load_dataset_for_run(
        model_type=model_type,
        dataset_path=dataset_path,
        config=config,
        summary=summary if isinstance(summary, dict) else {},
    )
    original_lookup = {} if model_type == 'meshcnn_full' else _load_original_dataset_lookup(dataset_path)
    split_info = dict(config.get('split') or {})
    resolution_tag = split_info.get('resolution_tag')
    if resolution_tag not in (None, '', 'all'):
        dataset = filter_dataset_by_resolution(dataset, resolution_tag)
    effective_limit = 1 if bool(getattr(args, 'debug_export', False)) else args.limit_meshes
    test_samples = _select_test_samples(dataset, split_info, effective_limit)

    model = predict_bridge.build_prediction_model(model_type, model_kwargs).to(device)
    state_dict = predict_bridge.extract_state_dict(predict_bridge.load_weights_payload(weights_path, device))
    state_dict_load = _load_model_state_for_evaluation(model, state_dict, model_type)
    model.eval()

    topology_cache: dict[str, Any] = {}
    raw_probabilities: list[np.ndarray] = []
    raw_labels: list[np.ndarray] = []
    post_masks: list[np.ndarray] = []
    per_mesh: list[dict[str, Any]] = []

    total_added_edges = 0
    total_pruned_edges = 0
    total_bridges = 0
    meshes_with_changes = 0
    debug_export_dir = (REPO_ROOT / 'debug_output') if bool(getattr(args, 'debug_export', False)) else None

    for sample in test_samples:
        mesh_path = _resolve_mesh_path(getattr(sample, 'file_path', ''))
        probabilities = _infer_probabilities(model_type, model, sample, device)

        if model_type == 'meshcnn_full':
            topology, _topology_edges = _load_topology(mesh_path, topology_cache)
            unique_edges = sample.unique_edges.detach().cpu().numpy().astype(np.int64)
            labels = sample.edge_labels.detach().cpu().numpy().astype(bool)
        else:
            topology = None
            original_sample = original_lookup.get(_sample_key(getattr(sample, 'file_path', '')))
            if original_sample is not None:
                unique_edges = _unique_edges_from_original_sample(original_sample)
                if debug_export_dir is not None:
                    topology, _topology_edges = _load_topology(mesh_path, topology_cache)
            else:
                topology, unique_edges = _load_topology(mesh_path, topology_cache)
            labels = sample.y.detach().cpu().numpy().astype(bool)

        if len(probabilities) != len(unique_edges):
            raise ValueError(
                f'{mesh_path.name}: probability count {len(probabilities)} does not match edge count {len(unique_edges)}'
            )
        if len(labels) != len(unique_edges):
            raise ValueError(
                f'{mesh_path.name}: label count {len(labels)} does not match edge count {len(unique_edges)}'
            )

        result = apply_seam_postprocessing_detailed(
            topology=topology,
            unique_edges=unique_edges,
            probabilities=probabilities,
            threshold=args.threshold,
            max_gap_length=args.max_gap_length,
            min_island_size=args.min_island_size,
            debug_export_dir=debug_export_dir,
        )
        raw_metrics = binary_metrics_from_probs(
            torch.from_numpy(probabilities.astype(np.float32)),
            torch.from_numpy(labels.astype(np.float32)),
            threshold=args.threshold,
        )
        post_metrics = _binary_metrics_from_mask(result.final_mask, labels)

        raw_mask = probabilities >= args.threshold
        if np.any(raw_mask != result.final_mask):
            meshes_with_changes += 1
        total_added_edges += len(result.steiner_added_edges)
        total_pruned_edges += len(result.pruned_edge_indices)
        total_bridges += int(result.steiner_tree_count)

        raw_probabilities.append(probabilities)
        raw_labels.append(labels.astype(np.float32))
        post_masks.append(result.final_mask.astype(bool))

        per_mesh.append({
            'file_path': str(mesh_path),
            'edge_count': int(len(unique_edges)),
            'gt_seam_count': int(np.count_nonzero(labels)),
            'raw_predicted_count': int(np.count_nonzero(raw_mask)),
            'post_predicted_count': int(np.count_nonzero(result.final_mask)),
            'raw_metrics': raw_metrics,
            'post_metrics': post_metrics,
            'delta': _metric_delta(post_metrics, raw_metrics),
            'postprocess': {
                'skeleton_deleted_vertices': [int(idx) for idx in result.skeleton_deleted_vertices],
                'skeleton_edge_count': int(np.count_nonzero(result.skeleton_mask)),
                'steiner_added_edges': [int(idx) for idx in result.steiner_added_edges],
                'pruned_edge_indices': [int(idx) for idx in result.pruned_edge_indices],
                'steiner_edge_count': int(np.count_nonzero(result.steiner_mask)),
                'skeleton_terminal_vertex_count': int(result.skeleton_terminal_vertex_count),
                'steiner_terminal_group_count': int(result.steiner_terminal_group_count),
                'steiner_tree_count': int(result.steiner_tree_count),
                'pruned_component_count': int(result.pruned_component_count),
            },
        })

    all_probs = np.concatenate(raw_probabilities) if raw_probabilities else np.zeros(0, dtype=np.float64)
    all_labels = np.concatenate(raw_labels) if raw_labels else np.zeros(0, dtype=np.float32)
    all_post = np.concatenate(post_masks) if post_masks else np.zeros(0, dtype=bool)
    raw_global = binary_metrics_from_probs(
        torch.from_numpy(all_probs.astype(np.float32)),
        torch.from_numpy(all_labels.astype(np.float32)),
        threshold=args.threshold,
    )
    post_global = _binary_metrics_from_mask(all_post, all_labels.astype(bool))

    return {
        'status': 'completed',
        'checkpoint': {
            'model_weights': str(weights_path),
            'config_json': str(config_path),
            'summary_json': str(summary_path),
            'dataset_path': str(dataset_path),
        },
        'model': {
            'model_type': model_type,
            'model_kwargs': model_kwargs,
            'device': str(device),
            'state_dict_load': state_dict_load,
        },
        'split': {
            'group_mode': split_info.get('group_mode'),
            'seed': split_info.get('seed'),
            'resolution_tag': resolution_tag,
            'test_group_ids': list(split_info.get('test') or ()),
            'evaluated_mesh_count': len(test_samples),
            'requested_mesh_path': None if args.mesh_path in (None, '') else str(_resolve_mesh_path(args.mesh_path)),
        },
        'postprocess': {
            'threshold': float(args.threshold),
            'max_gap_length': int(args.max_gap_length),
            'min_island_size': int(args.min_island_size),
            'debug_export': bool(getattr(args, 'debug_export', False)),
            'debug_export_dir': None if debug_export_dir is None else str(debug_export_dir.resolve()),
            'pipeline': (
                'local support smoothing, hysteresis thresholding, tiny component pruning, '
                'low-confidence spur pruning, terminal-set bridging with confidence-gated acceptance, final cleanup'
            ),
            'bridge_cost_function': '1 + alpha_cost * (1 - seam_probability)',
        },
        'global_metrics': {
            'pre_pp': raw_global,
            'post_pp': post_global,
            'delta': _metric_delta(post_global, raw_global),
        },
        'aggregate_postprocess': {
            'meshes_with_changes': int(meshes_with_changes),
            'total_added_edges': int(total_added_edges),
            'total_pruned_edges': int(total_pruned_edges),
            'total_bridges': int(total_bridges),
            'total_steiner_trees': int(total_bridges),
        },
        'per_mesh': per_mesh,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not hasattr(args, 'model_type'):
        args.model_type = 'auto'
    try:
        report = evaluate_checkpoint(args)
        output_json = (
            Path(args.output_json).resolve()
            if args.output_json
            else Path(args.model_weights).resolve().with_name('postprocess_eval.json')
        )
        _write_json(output_json, report)
        pre_metrics = report['global_metrics']['pre_pp']
        post_metrics = report['global_metrics']['post_pp']
        if pre_metrics is not None and post_metrics is not None:
            print(
                f"pre-pp f1 {pre_metrics['f1']:.4f} -> "
                f"post-pp f1 {post_metrics['f1']:.4f} "
                f"across {report['split']['evaluated_mesh_count']} mesh(es)"
            )
        else:
            print(
                f"processed {report['split']['evaluated_mesh_count']} standalone mesh(es) "
                f"without ground-truth metrics"
            )
        print(f'report written -> {output_json}')
        return 0
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
