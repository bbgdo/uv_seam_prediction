from __future__ import annotations

import itertools
from dataclasses import dataclass

from preprocessing.feature_registry import resolve_feature_selection


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    feature_group: str
    phase: str
    enable_ao: bool = False
    enable_dihedral: bool = False
    enable_symmetry: bool = False
    enable_density: bool = False
    enable_thickness_sdf: bool = False


BASELINE_EXPERIMENT = 'control14'
PHASE_BASELINE = 'base_baseline'
PHASE_ADD_ONE = 'add_one_in'
PHASE_PAIRWISE = 'pairwise_combinations'
FEATURE_TOKENS = ('ao', 'sdf', 'dihedral', 'symmetry', 'density')
DEFAULT_COMBINATORIAL_COUNTS = (1, 2)
VALID_COMBINATORIAL_COUNTS = tuple(range(1, len(FEATURE_TOKENS) + 1))
FLAG_BY_TOKEN = {
    'ao': 'enable_ao',
    'sdf': 'enable_thickness_sdf',
    'dihedral': 'enable_dihedral',
    'symmetry': 'enable_symmetry',
    'density': 'enable_density',
}
PAIRWISE_SEARCH_EXPERIMENTS_PER_ARCHITECTURE = 16
DEFAULT_SEED = 33
DEFAULT_EPOCHS = 60
DEFAULT_PATIENCE = 15
GNN_MODELS = ('graphsage', 'gatv2')
SPARSE_MESHCNN_MODEL = 'sparsemeshcnn'
ABLATION_MODELS = (*GNN_MODELS, SPARSE_MESHCNN_MODEL)


def _phase_for_custom_feature_count(count: int) -> str:
    if count == 1:
        return PHASE_ADD_ONE
    if count == 2:
        return PHASE_PAIRWISE
    return f'combinatorial_{count}'


def _build_experiment_specs(custom_feature_counts: tuple[int, ...]) -> dict[str, ExperimentSpec]:
    specs: dict[str, ExperimentSpec] = {
        BASELINE_EXPERIMENT: ExperimentSpec(
            name=BASELINE_EXPERIMENT,
            feature_group='paper14',
            phase=PHASE_BASELINE,
        ),
    }
    for size in sorted(set(custom_feature_counts)):
        for combo in itertools.combinations(FEATURE_TOKENS, size):
            name = '_'.join(combo)
            specs[name] = ExperimentSpec(
                name=name,
                feature_group='custom',
                phase=_phase_for_custom_feature_count(size),
                **{FLAG_BY_TOKEN[token]: True for token in combo},
            )
    return specs


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = _build_experiment_specs(DEFAULT_COMBINATORIAL_COUNTS)
ALL_EXPERIMENT_SPECS: dict[str, ExperimentSpec] = _build_experiment_specs(VALID_COMBINATORIAL_COUNTS)
FULL_ABLATION_SUITE = tuple(EXPERIMENT_SPECS)
ALL_COMBINATORIAL_SUITE = tuple(ALL_EXPERIMENT_SPECS)
if len(FULL_ABLATION_SUITE) != PAIRWISE_SEARCH_EXPERIMENTS_PER_ARCHITECTURE:
    raise RuntimeError('Pairwise Feature Search must define 16 experiments per architecture')


def combinatorial_suite(feature_counts: list[int] | tuple[int, ...]) -> list[str]:
    invalid = sorted(set(feature_counts) - set(VALID_COMBINATORIAL_COUNTS))
    if invalid:
        choices = ', '.join(str(value) for value in VALID_COMBINATORIAL_COUNTS)
        raise ValueError(f"invalid custom feature count(s) {invalid}; choose from: {choices}")
    return list(_build_experiment_specs(tuple(feature_counts)))


def get_experiment_spec(name: str) -> ExperimentSpec:
    try:
        return ALL_EXPERIMENT_SPECS[name]
    except KeyError as exc:
        choices = ', '.join(ALL_EXPERIMENT_SPECS)
        raise ValueError(f"unknown experiment {name!r}; choose one of: {choices}") from exc


def experiment_feature_selection(name: str):
    spec = get_experiment_spec(name)
    return resolve_feature_selection(
        spec.feature_group,
        enable_ao=spec.enable_ao,
        enable_dihedral=spec.enable_dihedral,
        enable_symmetry=spec.enable_symmetry,
        enable_density=spec.enable_density,
        enable_thickness_sdf=spec.enable_thickness_sdf,
    )


def experiment_feature_label(spec: ExperimentSpec) -> str:
    if spec.name == BASELINE_EXPERIMENT:
        return 'paper14'
    enabled = []
    if spec.enable_ao:
        enabled.append('ao')
    if spec.enable_thickness_sdf:
        enabled.append('sdf')
    if spec.enable_dihedral:
        enabled.append('dihedral')
    if spec.enable_symmetry:
        enabled.append('symmetry')
    if spec.enable_density:
        enabled.append('density')
    return 'paper14+' + '+'.join(enabled)


def is_meshcnn_model(model: str) -> bool:
    return model == SPARSE_MESHCNN_MODEL


def validate_experiment_selection(experiment_names: list[str], model: str) -> None:
    if model not in ABLATION_MODELS:
        choices = ', '.join(ABLATION_MODELS)
        raise ValueError(f"unsupported ablation model {model!r}; choose one of: {choices}")
    for name in experiment_names:
        get_experiment_spec(name)
