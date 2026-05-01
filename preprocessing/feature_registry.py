from dataclasses import dataclass


FEATURE_GROUP_NAMES = ('paper14', 'custom')

PAPER14_FEATURE_NAMES = (
    'pos_x_i', 'pos_y_i', 'pos_z_i',
    'normal_x_i', 'normal_y_i', 'normal_z_i',
    'gauss_curv_i',
    'pos_x_j', 'pos_y_j', 'pos_z_j',
    'normal_x_j', 'normal_y_j', 'normal_z_j',
    'gauss_curv_j',
)

_I_BASE = PAPER14_FEATURE_NAMES[:7]
_J_BASE = PAPER14_FEATURE_NAMES[7:]

AO_FEATURE_NAMES = ('ao_i', 'ao_j')
SIGNED_DIHEDRAL_FEATURE_NAMES = ('signed_dihedral',)
SYMMETRY_FEATURE_NAMES = ('symmetry_dist',)
DENSITY_FEATURE_NAMES = ('density_mean', 'density_diff')
THICKNESS_SDF_FEATURE_NAMES = ('thickness_sdf',)

ALL_ATOMIC_FEATURE_NAMES = (
    *_I_BASE,
    'ao_i',
    *_J_BASE,
    'ao_j',
    'signed_dihedral',
    'symmetry_dist',
    'density_mean',
    'density_diff',
    'thickness_sdf',
)

DENSITY_CONFIG = {
    'neighborhood': '2-ring',
    'support_area': 'one_third_incident_face_area',
    'eps': 1e-12,
    'density_log_clip': 3.0,
}


@dataclass(frozen=True)
class FeatureFlags:
    ao: bool = False
    signed_dihedral: bool = False
    symmetry: bool = False
    density: bool = False
    thickness_sdf: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            'ao': self.ao,
            'signed_dihedral': self.signed_dihedral,
            'symmetry': self.symmetry,
            'density': self.density,
            'thickness_sdf': self.thickness_sdf,
        }

    def any_enabled(self) -> bool:
        return any(self.as_dict().values())


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    feature_preset: str
    feature_names: tuple[str, ...]
    feature_flags: FeatureFlags

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class ResolvedFeatureSet:
    feature_group: str
    feature_preset: str
    feature_names: tuple[str, ...]
    feature_flags: FeatureFlags
    density_config: dict | None = None

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


FEATURE_GROUPS = {
    'paper14': FeatureGroup(
        name='paper14',
        feature_preset='paper14',
        feature_names=PAPER14_FEATURE_NAMES,
        feature_flags=FeatureFlags(),
    ),
    'custom': FeatureGroup(
        name='custom',
        feature_preset='custom',
        feature_names=PAPER14_FEATURE_NAMES,
        feature_flags=FeatureFlags(),
    ),
}


def _normalize_group_name(name: str | None) -> str:
    group = name or 'paper14'
    if group == 'paper':
        group = 'paper14'
    if group not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature group {group!r}; choose one of {FEATURE_GROUP_NAMES}")
    return group


def get_feature_group(name: str) -> FeatureGroup:
    return FEATURE_GROUPS[_normalize_group_name(name)]


def _custom_feature_names(flags: FeatureFlags) -> tuple[str, ...]:
    names: list[str] = list(_I_BASE)
    if flags.ao:
        names.append('ao_i')
    names.extend(_J_BASE)
    if flags.ao:
        names.append('ao_j')
    if flags.signed_dihedral:
        names.extend(SIGNED_DIHEDRAL_FEATURE_NAMES)
    if flags.symmetry:
        names.extend(SYMMETRY_FEATURE_NAMES)
    if flags.density:
        names.extend(DENSITY_FEATURE_NAMES)
    if flags.thickness_sdf:
        names.extend(THICKNESS_SDF_FEATURE_NAMES)
    return tuple(names)


def resolve_feature_selection(
    feature_group: str | None = None,
    *,
    enable_ao: bool = False,
    enable_dihedral: bool = False,
    enable_signed_dihedral: bool | None = None,
    enable_symmetry: bool = False,
    enable_density: bool = False,
    enable_thickness_sdf: bool = False,
) -> ResolvedFeatureSet:
    """Resolve a named bundle plus explicit extras into ordered feature names."""
    group_name = _normalize_group_name(feature_group)
    signed_dihedral = enable_dihedral if enable_signed_dihedral is None else enable_signed_dihedral
    requested_flags = FeatureFlags(
        ao=bool(enable_ao),
        signed_dihedral=bool(signed_dihedral),
        symmetry=bool(enable_symmetry),
        density=bool(enable_density),
        thickness_sdf=bool(enable_thickness_sdf),
    )

    if group_name != 'custom':
        if requested_flags.any_enabled():
            enabled = ', '.join(name for name, value in requested_flags.as_dict().items() if value)
            raise ValueError(
                f"feature toggles ({enabled}) require feature_group='custom'; "
                f"{group_name!r} is a locked bundle"
            )
        group = get_feature_group(group_name)
        density_config = dict(DENSITY_CONFIG) if group.feature_flags.density else None
        return ResolvedFeatureSet(
            feature_group=group.name,
            feature_preset=group.feature_preset,
            feature_names=group.feature_names,
            feature_flags=group.feature_flags,
            density_config=density_config,
        )

    names = _custom_feature_names(requested_flags)
    return ResolvedFeatureSet(
        feature_group='custom',
        feature_preset='custom',
        feature_names=names,
        feature_flags=requested_flags,
        density_config=dict(DENSITY_CONFIG) if requested_flags.density else None,
    )
