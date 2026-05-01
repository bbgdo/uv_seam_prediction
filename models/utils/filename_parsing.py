import re
from dataclasses import dataclass
from pathlib import Path
from typing import Pattern


DEFAULT_AUGMENTATION_PATTERN = r'_aug\d+$'
DEFAULT_RESOLUTION_PATTERNS = (
    r'_h$',
    r'_l$',
    r'_high$',
    r'_low$',
    r'_\d+f$',
    r'_res\d+$',
    r'_lod\d+$',
)


@dataclass(frozen=True)
class FilenameParseConfig:
    augmentation_pattern: str = DEFAULT_AUGMENTATION_PATTERN
    resolution_patterns: tuple[str, ...] = DEFAULT_RESOLUTION_PATTERNS


@dataclass(frozen=True)
class MeshNameInfo:
    stem: str
    family_id: str
    resolution_tag: str | None
    is_augmented: bool


def _strip_suffix(value: str, pattern: Pattern[str]) -> tuple[str, str | None]:
    match = pattern.search(value)
    if not match:
        return value, None

    token = match.group(0).lstrip('_').lower()
    return value[:match.start()], token


def _stem_from_path_or_name(path_or_name: str | Path) -> str:
    name = re.split(r'[\\/]', str(path_or_name))[-1]
    return Path(name).stem


def parse_mesh_name(path_or_name: str | Path, config: FilenameParseConfig | None = None) -> MeshNameInfo:
    """Parse mesh names for family-level dataset grouping."""
    config = config or FilenameParseConfig()
    stem = _stem_from_path_or_name(path_or_name)
    family_id = stem
    resolution_tag = None
    is_augmented = False

    aug_re = re.compile(config.augmentation_pattern, flags=re.IGNORECASE)
    resolution_res = [re.compile(pattern, flags=re.IGNORECASE) for pattern in config.resolution_patterns]

    previous = None
    while family_id and family_id != previous:
        previous = family_id

        family_id, aug_tag = _strip_suffix(family_id, aug_re)
        if aug_tag:
            is_augmented = True

        for pattern in resolution_res:
            stripped, tag = _strip_suffix(family_id, pattern)
            if tag:
                family_id = stripped
                resolution_tag = tag
                break

    return MeshNameInfo(
        stem=stem,
        family_id=family_id or stem,
        resolution_tag=resolution_tag,
        is_augmented=is_augmented,
    )
