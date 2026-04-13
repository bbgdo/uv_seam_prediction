from dataclasses import dataclass

try:
    from preprocessing.compute_features import EXTENDED18_FEATURE_NAMES, PAPER14_FEATURE_NAMES
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from compute_features import EXTENDED18_FEATURE_NAMES, PAPER14_FEATURE_NAMES


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    feature_preset: str
    feature_names: tuple[str, ...]


FEATURE_GROUPS = {
    'paper14': FeatureGroup(
        name='paper14',
        feature_preset='paper14',
        feature_names=tuple(PAPER14_FEATURE_NAMES),
    ),
    'extended18': FeatureGroup(
        name='extended18',
        feature_preset='extended18',
        feature_names=tuple(EXTENDED18_FEATURE_NAMES),
    ),
}


def get_feature_group(name: str) -> FeatureGroup:
    try:
        return FEATURE_GROUPS[name]
    except KeyError as exc:
        raise ValueError(f"unknown feature group {name!r}; choose one of {tuple(FEATURE_GROUPS)}") from exc
