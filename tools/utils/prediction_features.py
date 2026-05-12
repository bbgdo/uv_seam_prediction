from __future__ import annotations

import argparse
from typing import Any

from preprocessing.feature_registry import PAPER14_FEATURE_NAMES, ResolvedFeatureSet, resolve_feature_selection
from tools.utils.prediction_common import PredictionError, coerce_dict, coerce_list, normalize_metadata_name


def requested_feature_flags(args: argparse.Namespace) -> dict[str, bool]:
    return {
        'ao': bool(args.enable_ao),
        'signed_dihedral': bool(args.enable_dihedral),
        'symmetry': bool(args.enable_symmetry),
        'density': bool(args.enable_density),
        'thickness_sdf': bool(args.enable_thickness_sdf),
    }


def selection_from_feature_flags(flags: dict[str, bool]) -> ResolvedFeatureSet:
    if not any(flags.values()):
        return resolve_feature_selection('paper14')
    return resolve_feature_selection(
        'custom',
        enable_ao=flags['ao'],
        enable_dihedral=flags['signed_dihedral'],
        enable_symmetry=flags['symmetry'],
        enable_density=flags['density'],
        enable_thickness_sdf=flags['thickness_sdf'],
    )


def default_endpoint_order_for_selection(selection: ResolvedFeatureSet) -> str:
    return 'random' if selection.feature_group == 'paper14' else 'fixed'


def resolved_endpoint_order_from_metadata(
    metadata: dict[str, Any],
    selection: ResolvedFeatureSet,
) -> str:
    endpoint_order = normalize_metadata_name(metadata.get('endpoint_order'))
    if endpoint_order in ('fixed', 'random'):
        return endpoint_order
    group = normalize_metadata_name(metadata.get('feature_group'))
    if group == 'custom':
        return 'fixed'
    return default_endpoint_order_for_selection(selection)


def endpoint_order_from_metadata_sources(
    config: dict[str, Any],
    selection: ResolvedFeatureSet,
) -> str:
    for metadata in feature_metadata_sources(config):
        endpoint_order = normalize_metadata_name(metadata.get('endpoint_order'))
        if endpoint_order in ('fixed', 'random'):
            return endpoint_order
    return default_endpoint_order_for_selection(selection)


def resolve_feature_bundle(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[ResolvedFeatureSet, str, str]:
    flags = requested_feature_flags(args)
    any_toggle = any(flags.values())

    if args.feature_bundle == 'auto':
        if any_toggle:
            raise PredictionError(
                'feature toggles require an explicit --feature-bundle custom',
                'InvalidFeatureBundle',
            )
        return infer_feature_bundle(config)

    if args.feature_bundle != 'custom' and any_toggle:
        enabled = ', '.join(name for name, value in flags.items() if value)
        raise PredictionError(
            f'feature toggles ({enabled}) are only valid with --feature-bundle custom',
            'InvalidFeatureBundle',
        )

    if args.feature_bundle == 'paper14':
        selection = resolve_feature_selection('paper14')
        return selection, endpoint_order_from_metadata_sources(config, selection), args.feature_bundle

    if not any_toggle:
        raise PredictionError(
            '--feature-bundle custom requires at least one explicit feature toggle',
            'InvalidFeatureBundle',
        )
    selection = selection_from_feature_flags(flags)
    return selection, endpoint_order_from_metadata_sources(config, selection), args.feature_bundle


def infer_feature_bundle(config: dict[str, Any]) -> tuple[ResolvedFeatureSet, str, str]:
    for metadata in feature_metadata_sources(config):
        group = normalize_metadata_name(metadata.get('feature_group'))
        flags = infer_feature_flags(metadata)

        if group == 'paper14':
            selection = resolve_feature_selection('paper14')
            return selection, resolved_endpoint_order_from_metadata(metadata, selection), 'auto'
        if group == 'custom':
            if not any(flags.values()):
                raise PredictionError(
                    'feature metadata declares custom features but does not specify any optional custom feature flags',
                    'MissingFeatureMetadata',
                )
            selection = selection_from_feature_flags(flags)
            return selection, resolved_endpoint_order_from_metadata(metadata, selection), 'auto'

    for metadata in feature_metadata_sources(config):
        feature_names = coerce_list(metadata.get('feature_names'))
        if feature_names:
            names = tuple(feature_names)
            if names == PAPER14_FEATURE_NAMES:
                selection = resolve_feature_selection('paper14')
                return selection, resolved_endpoint_order_from_metadata(metadata, selection), 'auto'
            flags = infer_feature_flags({'feature_names': feature_names})
            if not any(flags.values()):
                raise PredictionError(
                    f'feature_names do not match paper14 and do not identify a supported custom feature set: {feature_names}',
                    'MissingFeatureMetadata',
                )
            selection = selection_from_feature_flags(flags)
            return selection, resolved_endpoint_order_from_metadata(metadata, selection), 'auto'

    raise PredictionError(
        'feature bundle could not be inferred from config metadata; pass --feature-bundle explicitly',
        'MissingFeatureMetadata',
    )


def feature_metadata_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [config]
    feature_metadata = coerce_dict(config.get('feature_metadata'))
    if 'feature_metadata' in config and config.get('feature_metadata') not in (None, '') and feature_metadata is None:
        raise PredictionError('config feature_metadata must be a JSON object', 'FeatureMetadataMismatch')
    if feature_metadata is not None:
        sources.append(feature_metadata)
    return sources


def infer_feature_flags(metadata: dict[str, Any]) -> dict[str, bool]:
    flags = coerce_dict(metadata.get('feature_flags'))
    if 'feature_flags' in metadata and metadata.get('feature_flags') not in (None, '') and flags is None:
        raise PredictionError('feature_flags must be a JSON object', 'FeatureMetadataMismatch')
    feature_names = coerce_list(metadata.get('feature_names'))
    if 'feature_names' in metadata and metadata.get('feature_names') not in (None, '') and feature_names is None:
        raise PredictionError('feature_names must be a JSON list', 'FeatureMetadataMismatch')
    flags = flags or {}
    names = set(feature_names or ())
    return {
        'ao': bool(flags.get('ao')) or 'ao_i' in names or 'ao_j' in names,
        'signed_dihedral': (
            bool(flags.get('signed_dihedral'))
            or 'signed_dihedral' in names
        ),
        'symmetry': bool(flags.get('symmetry')) or 'symmetry_dist' in names,
        'density': bool(flags.get('density')) or 'density_mean' in names or 'density_diff' in names,
        'thickness_sdf': bool(flags.get('thickness_sdf')) or 'thickness_sdf' in names,
    }


def validate_feature_metadata(
    config: dict[str, Any],
    selection: ResolvedFeatureSet,
    model_kwargs: dict[str, Any],
) -> None:
    sources = [
        ('config', config),
    ]
    feature_metadata = coerce_dict(config.get('feature_metadata'))
    if 'feature_metadata' in config and config.get('feature_metadata') not in (None, '') and feature_metadata is None:
        raise PredictionError('config feature_metadata must be a JSON object', 'FeatureMetadataMismatch')
    if feature_metadata is not None:
        sources.append(('config.feature_metadata', feature_metadata))

    expected_flags = selection.feature_flags.as_dict()
    for source_name, metadata in sources:
        validate_feature_metadata_name(source_name, metadata, selection)

        feature_names = coerce_list(metadata.get('feature_names'))
        if 'feature_names' in metadata and metadata.get('feature_names') not in (None, '') and feature_names is None:
            raise PredictionError(
                f'{source_name} feature_names must be a JSON list',
                'FeatureMetadataMismatch',
            )
        if feature_names is not None and feature_names != list(selection.feature_names):
            raise PredictionError(
                f'{source_name} feature_names mismatch: expected {list(selection.feature_names)}, got {feature_names}',
                'FeatureMetadataMismatch',
            )

        flags = coerce_dict(metadata.get('feature_flags'))
        if 'feature_flags' in metadata and metadata.get('feature_flags') not in (None, '') and flags is None:
            raise PredictionError(
                f'{source_name} feature_flags must be a JSON object',
                'FeatureMetadataMismatch',
            )
        if flags is not None:
            unknown_flags = sorted(set(flags) - set(expected_flags))
            if unknown_flags:
                raise PredictionError(
                    f'{source_name} feature_flags contains unsupported key(s): {unknown_flags}',
                    'FeatureMetadataMismatch',
                )
            for key, expected_value in expected_flags.items():
                if key in flags and bool(flags[key]) != bool(expected_value):
                    raise PredictionError(
                        f'{source_name} feature_flags mismatch for {key}: '
                        f'expected {expected_value}, got {flags[key]}',
                        'FeatureMetadataMismatch',
                    )

        dim_key = None
        for candidate_key in ('in_dim', 'feature_dim', 'in_channels'):
            if candidate_key in metadata and metadata.get(candidate_key) not in (None, ''):
                dim_key = candidate_key
                break
        if dim_key is not None:
            observed = int(metadata[dim_key])
            if observed != selection.feature_count:
                raise PredictionError(
                    f'{source_name} {dim_key} mismatch: selected features={selection.feature_count}, metadata={observed}',
                    'FeatureMetadataMismatch',
                )

    model_in_dim = int(model_kwargs.get('in_dim', model_kwargs.get('in_channels')))
    if model_in_dim != selection.feature_count:
        raise PredictionError(
            f'model in_dim mismatch: selected features={selection.feature_count}, model in_dim={model_in_dim}',
            'FeatureMetadataMismatch',
        )


def validate_feature_metadata_name(
    source_name: str,
    metadata: dict[str, Any],
    selection: ResolvedFeatureSet,
) -> None:
    value = metadata.get('feature_group')
    if value in (None, ''):
        return
    if metadata_name_matches_expected(value, selection.feature_group):
        return
    raise PredictionError(
        f"{source_name} feature_group mismatch: expected {selection.feature_group!r}, got {value!r}",
        'FeatureMetadataMismatch',
    )


def metadata_name_matches_expected(value: Any, expected: str) -> bool:
    if isinstance(value, (list, tuple, set)):
        values = {normalize_metadata_name(item) for item in value if item not in (None, '')}
        return normalize_metadata_name(expected) in values
    return normalize_metadata_name(value) == normalize_metadata_name(expected)
