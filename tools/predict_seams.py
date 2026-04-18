from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import trimesh  # noqa: E402
from models.baselines.registry import get_baseline  # noqa: E402
from preprocessing.build_dual_graph import build_dual_edge_index_from_unique_edges  # noqa: E402
from preprocessing.compute_features import compute_edge_features_for_selection  # noqa: E402
from preprocessing.feature_registry import ResolvedFeatureSet, resolve_feature_selection  # noqa: E402
from preprocessing.obj_parser import parse_obj  # noqa: E402
from preprocessing.topology import CanonicalTopology, WeldConfig, build_topology  # noqa: E402


MODEL_TYPES = ('auto', 'gatv2', 'graphsage')
FEATURE_BUNDLES = ('auto', 'paper14_locked', 'extended18', 'ao_density', 'custom')


class PredictionError(RuntimeError):
    def __init__(self, message: str, error_type: str = 'PredictionError'):
        super().__init__(message)
        self.error_type = error_type


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict UV seam edges for a raw OBJ mesh.')
    parser.add_argument('--mesh-path', required=True)
    parser.add_argument('--model-weights', required=True)
    parser.add_argument('--feature-bundle', default='auto', choices=FEATURE_BUNDLES)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--model-type', choices=MODEL_TYPES, default='auto')
    parser.add_argument('--config-json', default=None)
    parser.add_argument('--summary-json', default=None)
    parser.add_argument('--enable-ao', action='store_true')
    parser.add_argument('--enable-dihedral', action='store_true')
    parser.add_argument('--enable-symmetry', action='store_true')
    parser.add_argument('--enable-density', action='store_true')
    parser.add_argument('--endpoint-seed', type=int, default=42)
    parser.add_argument('--write-all-edges', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fail-if-threshold-missing', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise PredictionError(f'{label} is not valid JSON: {path}', 'InvalidJson') from exc
    if not isinstance(payload, dict):
        raise PredictionError(f'{label} must contain a JSON object: {path}', 'InvalidJson')
    return payload


def resolve_threshold(
    explicit_threshold: float | None,
    summary: dict[str, Any],
    fail_if_missing: bool = True,
) -> float:
    if explicit_threshold is not None:
        return _validate_threshold(explicit_threshold)

    if 'best_validation_threshold' in summary:
        return _validate_threshold(summary['best_validation_threshold'])

    suffix = ''
    if not fail_if_missing:
        suffix = '; no alternate threshold policy is implemented'
    raise PredictionError(
        'threshold is required: pass --threshold or provide summary.json["best_validation_threshold"]' + suffix,
        'MissingThreshold',
    )


def _validate_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionError(f'threshold must be a number, got {value!r}', 'InvalidThreshold') from exc
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise PredictionError(f'threshold must be a finite value in [0, 1], got {threshold}', 'InvalidThreshold')
    return threshold


def resolve_model_type(requested: str, config: dict[str, Any], weights_path: Path) -> str:
    if requested != 'auto':
        return requested

    for key in ('model', 'model_name'):
        resolved = _normalize_model_name(config.get(key))
        if resolved is not None:
            return resolved

    resolved = _normalize_model_name(weights_path.parent.name)
    if resolved is not None:
        return resolved

    raise PredictionError(
        'model type could not be resolved from --model-type, config metadata, or parent run directory',
        'MissingModelType',
    )


def _normalize_model_name(value: Any) -> str | None:
    if value in (None, ''):
        return None
    normalized = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    if normalized == 'gatv2' or 'gatv2' in normalized:
        return 'gatv2'
    if normalized == 'graphsage' or 'graphsage' in normalized:
        return 'graphsage'
    return None


def resolve_feature_bundle(
    args: argparse.Namespace,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[ResolvedFeatureSet, str, str]:
    toggles = {
        'enable_ao': bool(args.enable_ao),
        'enable_dihedral': bool(args.enable_dihedral),
        'enable_symmetry': bool(args.enable_symmetry),
        'enable_density': bool(args.enable_density),
    }
    any_toggle = any(toggles.values())

    if args.feature_bundle == 'auto':
        if any_toggle:
            raise PredictionError(
                'feature toggles require an explicit --feature-bundle custom',
                'InvalidFeatureBundle',
            )
        return infer_feature_bundle(config, summary)

    if args.feature_bundle != 'custom' and any_toggle:
        enabled = ', '.join(name for name, value in toggles.items() if value)
        raise PredictionError(
            f'feature toggles ({enabled}) are only valid with --feature-bundle custom',
            'InvalidFeatureBundle',
        )

    if args.feature_bundle == 'paper14_locked':
        return resolve_feature_selection('paper14'), 'random', args.feature_bundle
    if args.feature_bundle == 'extended18':
        return resolve_feature_selection('extended18'), 'fixed', args.feature_bundle
    if args.feature_bundle == 'ao_density':
        return resolve_feature_selection('custom', enable_ao=True, enable_density=True), 'fixed', args.feature_bundle

    if not any_toggle:
        raise PredictionError(
            '--feature-bundle custom requires at least one explicit feature toggle',
            'InvalidFeatureBundle',
        )
    return (
        resolve_feature_selection(
            'custom',
            enable_ao=args.enable_ao,
            enable_dihedral=args.enable_dihedral,
            enable_symmetry=args.enable_symmetry,
            enable_density=args.enable_density,
        ),
        'fixed',
        args.feature_bundle,
    )


def infer_feature_bundle(config: dict[str, Any], summary: dict[str, Any]) -> tuple[ResolvedFeatureSet, str, str]:
    for metadata in _feature_metadata_sources(config, summary):
        group = _normalize_metadata_name(metadata.get('feature_group'))
        preset = _normalize_metadata_name(metadata.get('feature_preset'))

        if group in ('paper14', 'paper') or preset in ('paper14', 'paper'):
            return resolve_feature_selection('paper14'), 'random', 'auto'
        if group in ('extended18', 'extended') or preset in ('extended18', 'extended'):
            return resolve_feature_selection('extended18'), 'fixed', 'auto'
        if group == 'custom' or preset == 'custom':
            flags = _infer_feature_flags(metadata)
            return (
                resolve_feature_selection(
                    'custom',
                    enable_ao=flags['ao'],
                    enable_signed_dihedral=flags['signed_dihedral'],
                    enable_symmetry=flags['symmetry'],
                    enable_density=flags['density'],
                    enable_thickness_sdf=flags['thickness_sdf'],
                ),
                'fixed',
                'auto',
            )

    for metadata in _feature_metadata_sources(config, summary):
        feature_names = _coerce_list(metadata.get('feature_names'))
        if feature_names:
            names = tuple(feature_names)
            if names == resolve_feature_selection('paper14').feature_names:
                return resolve_feature_selection('paper14'), 'random', 'auto'
            if names == resolve_feature_selection('extended18').feature_names:
                return resolve_feature_selection('extended18'), 'fixed', 'auto'
            flags = _infer_feature_flags({'feature_names': feature_names})
            return (
                resolve_feature_selection(
                    'custom',
                    enable_ao=flags['ao'],
                    enable_signed_dihedral=flags['signed_dihedral'],
                    enable_symmetry=flags['symmetry'],
                    enable_density=flags['density'],
                    enable_thickness_sdf=flags['thickness_sdf'],
                ),
                'fixed',
                'auto',
            )

    return resolve_feature_selection('custom', enable_ao=True, enable_density=True), 'fixed', 'auto'


def _feature_metadata_sources(config: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [config, summary]
    dataset_summary = summary.get('dataset_metadata_summary')
    if isinstance(dataset_summary, dict):
        sources.append(dataset_summary)
    return sources


def _normalize_metadata_name(value: Any) -> str | None:
    if value in (None, ''):
        return None
    return str(value).strip().lower().replace('-', '_').replace(' ', '_')


def _infer_feature_flags(metadata: dict[str, Any]) -> dict[str, bool]:
    flags = _coerce_dict(metadata.get('feature_flags')) or {}
    names = set(_coerce_list(metadata.get('feature_names')) or ())
    return {
        'ao': bool(flags.get('ao')) or 'ao_i' in names or 'ao_j' in names,
        'signed_dihedral': (
            bool(flags.get('signed_dihedral'))
            or bool(flags.get('dihedral'))
            or 'signed_dihedral' in names
        ),
        'symmetry': bool(flags.get('symmetry')) or 'symmetry_dist' in names,
        'density': bool(flags.get('density')) or 'density_mean' in names or 'density_diff' in names,
        'thickness_sdf': bool(flags.get('thickness_sdf')) or 'thickness_sdf' in names,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise PredictionError('requested CUDA device, but CUDA is unavailable', 'UnavailableDevice')
    return torch.device(requested)


def resolve_model_kwargs(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    in_dim = _required_config_value(config, ('in_dim',), 'in_dim')
    hidden_dim = _required_config_value(config, ('hidden_size', 'hidden_dim', 'hidden'), 'hidden_size/hidden_dim/hidden')
    kwargs = {
        'in_dim': int(in_dim),
        'hidden_dim': int(hidden_dim),
        'num_layers': int(_required_config_value(config, ('num_layers',), 'num_layers')),
        'dropout': float(_required_config_value(config, ('dropout',), 'dropout')),
    }
    if model_name == 'gatv2':
        kwargs['heads'] = int(_required_config_value(config, ('heads',), 'heads'))
    elif model_name == 'graphsage':
        kwargs['aggr'] = str(_required_config_value(config, ('aggr',), 'aggr'))
        kwargs['skip_connections'] = str(
            _required_config_value(config, ('skip_connections',), 'skip_connections')
        )
    else:
        raise PredictionError(f'unsupported model type: {model_name}', 'InvalidModelType')
    return kwargs


def _required_config_value(config: dict[str, Any], keys: tuple[str, ...], label: str) -> Any:
    for key in keys:
        if key in config and config[key] not in (None, ''):
            return config[key]
    raise PredictionError(f'config metadata is missing required model key: {label}', 'InvalidConfig')


def validate_feature_metadata(
    config: dict[str, Any],
    summary: dict[str, Any],
    selection: ResolvedFeatureSet,
    model_kwargs: dict[str, Any],
) -> None:
    sources = [
        ('config', config),
        ('summary', summary),
    ]
    dataset_summary = summary.get('dataset_metadata_summary')
    if isinstance(dataset_summary, dict):
        sources.append(('summary.dataset_metadata_summary', dataset_summary))

    expected_flags = selection.feature_flags.as_dict()
    for source_name, metadata in sources:
        _validate_metadata_scalar(source_name, metadata, 'feature_group', selection.feature_group)
        _validate_metadata_scalar(source_name, metadata, 'feature_preset', selection.feature_preset)

        feature_names = _coerce_list(metadata.get('feature_names'))
        if feature_names is not None and feature_names != list(selection.feature_names):
            raise PredictionError(
                f'{source_name} feature_names mismatch: expected {list(selection.feature_names)}, got {feature_names}',
                'FeatureMetadataMismatch',
            )

        flags = _coerce_dict(metadata.get('feature_flags'))
        if flags is not None:
            for key, expected_value in expected_flags.items():
                aliases = (key,)
                if key == 'signed_dihedral':
                    aliases = ('signed_dihedral', 'dihedral')
                present = [alias for alias in aliases if alias in flags]
                if present and bool(flags[present[0]]) != bool(expected_value):
                    raise PredictionError(
                        f'{source_name} feature_flags mismatch for {present[0]}: '
                        f'expected {expected_value}, got {flags[present[0]]}',
                        'FeatureMetadataMismatch',
                    )

        if 'in_dim' in metadata and metadata.get('in_dim') not in (None, ''):
            observed = int(metadata['in_dim'])
            if observed != selection.feature_count:
                raise PredictionError(
                    f'{source_name} in_dim mismatch: selected features={selection.feature_count}, metadata={observed}',
                    'FeatureMetadataMismatch',
                )

    if int(model_kwargs['in_dim']) != selection.feature_count:
        raise PredictionError(
            f'model in_dim mismatch: selected features={selection.feature_count}, model in_dim={model_kwargs["in_dim"]}',
            'FeatureMetadataMismatch',
        )


def _validate_metadata_scalar(source_name: str, metadata: dict[str, Any], key: str, expected: str) -> None:
    value = metadata.get(key)
    if value in (None, ''):
        return
    if isinstance(value, (list, tuple, set)):
        values = {str(item) for item in value if item not in (None, '')}
        if values and expected not in values:
            raise PredictionError(
                f'{source_name} {key} mismatch: expected {expected!r}, got {sorted(values)}',
                'FeatureMetadataMismatch',
            )
        return
    if str(value) != expected:
        raise PredictionError(
            f'{source_name} {key} mismatch: expected {expected!r}, got {value!r}',
            'FeatureMetadataMismatch',
        )


def _coerce_list(value: Any) -> list[str] | None:
    if value in (None, ''):
        return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        value = parsed
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    if value in (None, ''):
        return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        value = parsed
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return None


def build_feature_mesh_from_canonical_topology(topology: CanonicalTopology) -> trimesh.Trimesh:
    vertices = np.asarray(topology.canonical_vertices, dtype=np.float64)
    faces = np.asarray([face.vertex_ids for face in topology.canonical_faces], dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise PredictionError('input OBJ produced an empty feature mesh', 'InvalidMesh')
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def assert_canonical_edge_order(unique_edges: np.ndarray, canonical_edges: tuple, mesh_path: Path) -> None:
    expected_edges = np.asarray(canonical_edges, dtype=np.int64).reshape((-1, 2))
    if np.array_equal(unique_edges, expected_edges):
        return

    if unique_edges.shape != expected_edges.shape:
        detail = f'feature_edges shape={unique_edges.shape}, canonical_edges shape={expected_edges.shape}'
    else:
        mismatch_rows = np.flatnonzero(np.any(unique_edges != expected_edges, axis=1))
        idx = int(mismatch_rows[0]) if len(mismatch_rows) else -1
        detail = f'first mismatch at {idx}: feature={tuple(unique_edges[idx])}, canonical={tuple(expected_edges[idx])}'
    raise PredictionError(
        f'canonical edge order mismatch for {mesh_path}: {detail}',
        'EdgeOrderMismatch',
    )


def build_dual_data(edge_features: np.ndarray, unique_edges: np.ndarray) -> Data:
    dual_edge_index = build_dual_edge_index_from_unique_edges(unique_edges)
    dual_x = torch.from_numpy(edge_features).float()
    return Data(x=dual_x, edge_index=dual_edge_index, num_nodes=len(unique_edges))


def normalize_probabilities(probabilities: np.ndarray, expected_length: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape == (expected_length,):
        return probs.astype(float)
    if probs.shape == (expected_length, 1):
        return probs[:, 0].astype(float)
    if probs.shape == (1, expected_length):
        return probs[0].astype(float)
    raise PredictionError(
        f'model output shape {probs.shape} cannot be normalized to {expected_length} edge probabilities',
        'InvalidModelOutput',
    )


def load_state_dict(weights_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        try:
            payload = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            payload = torch.load(weights_path, map_location=device)
    except Exception as exc:
        raise PredictionError(f'failed to load model weights: {weights_path}', 'InvalidWeights') from exc

    if isinstance(payload, dict):
        for key in ('state_dict', 'model_state_dict'):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return nested
        if all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise PredictionError(
        'model weights must be a state_dict or contain state_dict/model_state_dict',
        'InvalidWeights',
    )


def build_output_payload(
    *,
    mesh_path: Path,
    output_json: Path,
    weights_path: Path,
    config_path: Path,
    summary_path: Path,
    model_type: str,
    feature_bundle: str,
    selection: ResolvedFeatureSet,
    threshold: float,
    device: torch.device,
    topology: CanonicalTopology,
    unique_edges: np.ndarray,
    probabilities: np.ndarray,
    seam_mask: np.ndarray,
    write_all_edges: bool,
) -> dict[str, Any]:
    seam_edge_indices = [int(idx) for idx in np.flatnonzero(seam_mask)]
    edge_rows = []
    seam_rows = []
    boundary_count = 0

    for idx, edge in enumerate(unique_edges):
        vi, vj = int(edge[0]), int(edge[1])
        edge_key = (vi, vj)
        is_boundary = len(topology.edge_incidence[edge_key]) == 1
        if is_boundary:
            boundary_count += 1
        row = {
            'canonical_edge_index': int(idx),
            'vertex_ids_0based': [vi, vj],
            'vertex_ids_obj_1based': [vi + 1, vj + 1],
            'probability': float(probabilities[idx]),
            'predicted_seam': bool(seam_mask[idx]),
            'is_boundary': bool(is_boundary),
        }
        if write_all_edges:
            edge_rows.append(row)
        if seam_mask[idx]:
            seam_rows.append({
                'canonical_edge_index': row['canonical_edge_index'],
                'vertex_ids_0based': row['vertex_ids_0based'],
                'vertex_ids_obj_1based': row['vertex_ids_obj_1based'],
                'probability': row['probability'],
                'is_boundary': row['is_boundary'],
            })

    payload: dict[str, Any] = {
        'schema_version': 1,
        'status': 'ok',
        'mesh_path': str(mesh_path.resolve()),
        'output_json': str(output_json.resolve()),
        'model': {
            'model_type': model_type,
            'weights_path': str(weights_path.resolve()),
            'config_path': str(config_path.resolve()),
            'summary_path': str(summary_path.resolve()),
            'feature_bundle': feature_bundle,
            'feature_group': selection.feature_group,
            'feature_names': list(selection.feature_names),
            'in_dim': int(selection.feature_count),
            'threshold': float(threshold),
            'device': str(device),
        },
        'topology': {
            'vertex_count': int(len(topology.canonical_vertices)),
            'face_count': int(len(topology.canonical_faces)),
            'edge_count': int(len(unique_edges)),
            'edge_order': 'canonical_topology_sorted_edge_keys',
        },
        'stats': {
            'predicted_seam_count': int(len(seam_edge_indices)),
            'boundary_edge_count': int(boundary_count),
            'probability_min': float(np.min(probabilities)) if len(probabilities) else 0.0,
            'probability_max': float(np.max(probabilities)) if len(probabilities) else 0.0,
            'probability_mean': float(np.mean(probabilities)) if len(probabilities) else 0.0,
        },
        'seam_edge_indices': seam_edge_indices,
        'seam_edges': seam_rows,
    }
    if write_all_edges:
        payload['edges'] = edge_rows
    return payload


def write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')


def write_error_payload(path: Path, error_type: str, message: str) -> None:
    payload = {
        'schema_version': 1,
        'status': 'error',
        'error_type': error_type,
        'message': message,
    }
    write_json_payload(path, payload)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PredictionError(f'{label} not found: {path}', 'MissingFile')


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    mesh_path = Path(args.mesh_path)
    weights_path = Path(args.model_weights)
    output_json = Path(args.output_json)
    config_path = Path(args.config_json) if args.config_json else weights_path.with_name('config.json')
    summary_path = Path(args.summary_json) if args.summary_json else weights_path.with_name('summary.json')

    _require_file(mesh_path, 'input OBJ')
    _require_file(weights_path, 'model weights')
    _require_file(config_path, 'config JSON')
    _require_file(summary_path, 'summary JSON')

    config = load_json(config_path, 'config JSON')
    summary = load_json(summary_path, 'summary JSON')
    model_type = resolve_model_type(args.model_type, config, weights_path)
    threshold = resolve_threshold(args.threshold, summary, args.fail_if_threshold_missing)
    selection, endpoint_order, resolved_feature_bundle = resolve_feature_bundle(args, config, summary)
    device = resolve_device(args.device)
    model_kwargs = resolve_model_kwargs(model_type, config)
    validate_feature_metadata(config, summary, selection, model_kwargs)

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

    dual_data = build_dual_data(edge_features, unique_edges)
    definition = get_baseline(model_type)
    model = definition.model_class(**model_kwargs)
    state_dict = load_state_dict(weights_path, device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise PredictionError(f'model state dict did not load cleanly: {exc}', 'InvalidWeights') from exc
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(dual_data.x.to(device), dual_data.edge_index.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    probabilities = normalize_probabilities(probs, len(unique_edges))
    seam_mask = probabilities >= threshold

    return build_output_payload(
        mesh_path=mesh_path,
        output_json=output_json,
        weights_path=weights_path,
        config_path=config_path,
        summary_path=summary_path,
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
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = Path(args.output_json)
    try:
        payload = run_prediction(args)
        write_json_payload(output_json, payload)
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
