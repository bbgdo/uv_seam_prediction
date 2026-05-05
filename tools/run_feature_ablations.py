import argparse
import csv
import itertools
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.utils.dataset import (  # noqa: E402
    filter_dataset_by_resolution,
    load_dataset,
    load_split_json_metadata,
    split_dataset,
)
from models.meshcnn_full.mesh import load_meshcnn_dataset  # noqa: E402
from preprocessing.feature_registry import resolve_feature_selection  # noqa: E402


METRIC_KEYS = ('f1', 'precision', 'recall', 'fpr', 'tpr', 'accuracy')
DELTA_METRIC_KEYS = ('fpr', 'recall', 'f1', 'accuracy')
THRESHOLD_05_PREFIX = 'test_0_5'
VAL_BEST_PREFIX = 'test_val_best'
GNN_MODELS = ('graphsage', 'gatv2')
SPARSE_MESHCNN_MODEL = 'sparsemeshcnn'
SPARSE_MESHCNN_TRAIN_SCRIPT = Path('models') / 'meshcnn_full' / 'train.py'
ABLATION_MODELS = (*GNN_MODELS, SPARSE_MESHCNN_MODEL)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    feature_group: str
    enable_ao: bool = False
    enable_dihedral: bool = False
    enable_symmetry: bool = False
    enable_density: bool = False
    enable_thickness_sdf: bool = False


_FEATURE_TOKENS = ('ao', 'dihedral', 'symmetry', 'density', 'sdf')
_FLAG_KEYS = ('enable_ao', 'enable_dihedral', 'enable_symmetry', 'enable_density', 'enable_thickness_sdf')


def _build_experiment_specs() -> dict[str, ExperimentSpec]:
    specs: dict[str, ExperimentSpec] = {
        'control14': ExperimentSpec(name='control14', feature_group='custom'),
    }
    for size in range(1, len(_FEATURE_TOKENS) + 1):
        for combo in itertools.combinations(range(len(_FEATURE_TOKENS)), size):
            name = '_'.join(_FEATURE_TOKENS[i] for i in combo)
            specs[name] = ExperimentSpec(
                name=name,
                feature_group='custom',
                **{_FLAG_KEYS[i]: True for i in combo},
            )
    return specs


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = _build_experiment_specs()
FULL_ABLATION_SUITE = tuple(EXPERIMENT_SPECS)


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


def get_experiment_spec(name: str) -> ExperimentSpec:
    try:
        return EXPERIMENT_SPECS[name]
    except KeyError as exc:
        choices = ', '.join(EXPERIMENT_SPECS)
        raise ValueError(f"unknown experiment {name!r}; choose one of: {choices}") from exc


def is_meshcnn_model(model: str) -> bool:
    return model == SPARSE_MESHCNN_MODEL


def validate_experiment_selection(experiment_names: list[str], model: str = 'graphsage') -> None:
    if model not in ABLATION_MODELS:
        choices = ', '.join(ABLATION_MODELS)
        raise ValueError(f"unsupported ablation model {model!r}; choose one of: {choices}")
    for name in experiment_names:
        get_experiment_spec(name)


def split_path_for_seed(splits_dir: Path, seed: int) -> Path:
    return splits_dir / f'seed_{seed}.json'


def _metric_columns(prefix: str) -> list[str]:
    return [f'{prefix}_{metric}' for metric in METRIC_KEYS]


def _delta_columns(prefix: str) -> list[str]:
    return [f'delta_{prefix}_{metric}' for metric in DELTA_METRIC_KEYS]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _metadata_value(data, key: str):
    try:
        value = getattr(data, key)
        if value not in (None, ''):
            return value
    except AttributeError:
        pass

    for container_key in ('metadata', 'meta', 'dataset_metadata'):
        try:
            container = getattr(data, container_key)
        except AttributeError:
            continue
        if isinstance(container, dict) and container.get(key) not in (None, ''):
            return container[key]
        if hasattr(container, key):
            value = getattr(container, key)
            if value not in (None, ''):
                return value
    return None


def _unique_string_values(dataset: list, key: str) -> tuple[list[str], int]:
    values = []
    missing = 0
    for data in dataset:
        value = _metadata_value(data, key)
        if value in (None, ''):
            missing += 1
        else:
            values.append(str(value))
    return sorted(set(values)), missing


def _require_uniform_metadata(dataset: list, *, role: str, key: str, expected: str) -> None:
    observed, missing = _unique_string_values(dataset, key)
    if missing or observed != [expected]:
        detail = f"observed={observed or 'none'}"
        if missing:
            detail += f", missing={missing}"
        raise ValueError(f"{role} dataset {key} must be {expected!r} ({detail})")


def _coerce_feature_names(value) -> list[str] | None:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def validate_custom_dataset_metadata(dataset: list, experiment_names: list[str]) -> None:
    if not dataset:
        raise ValueError('custom dataset is empty after resolution filtering')
    _require_uniform_metadata(dataset, role='custom', key='feature_group', expected='custom')
    _require_uniform_metadata(dataset, role='custom', key='endpoint_order', expected='random')

    requested_features: list[str] = []
    for name in experiment_names:
        for feature_name in experiment_feature_selection(name).feature_names:
            if feature_name not in requested_features:
                requested_features.append(feature_name)

    if not requested_features:
        return

    for graph_idx, data in enumerate(dataset):
        names = _coerce_feature_names(_metadata_value(data, 'feature_names'))
        if names is None:
            raise ValueError(f"custom dataset graph {graph_idx} is missing feature_names metadata")
        x = getattr(data, 'x', None)
        if x is not None and len(names) != int(x.shape[1]):
            raise ValueError(
                f"custom dataset graph {graph_idx} feature_names length {len(names)} "
                f"does not match x feature dim {int(x.shape[1])}"
            )
        missing = [feature for feature in requested_features if feature not in names]
        if missing:
            raise ValueError(
                f"custom dataset graph {graph_idx} is missing requested feature(s): {missing}; "
                f"available feature_names={names}"
            )


def validate_meshcnn_dataset_metadata(dataset: list, experiment_names: list[str]) -> None:
    if not dataset:
        raise ValueError('MeshCNN dataset is empty after resolution filtering')
    _require_uniform_metadata(dataset, role='MeshCNN', key='endpoint_order', expected='random')

    label_sources, missing_label_source = _unique_string_values(dataset, 'label_source')
    if label_sources and label_sources != ['exact_obj']:
        detail = f"observed={label_sources}"
        if missing_label_source:
            detail += f", missing={missing_label_source}"
        raise ValueError(f"MeshCNN dataset label_source must be 'exact_obj' when present ({detail})")

    available_names = _coerce_feature_names(_metadata_value(dataset[0], 'feature_names'))
    if available_names is None:
        raise ValueError('MeshCNN dataset sample 0 is missing feature_names metadata')

    for sample_idx, sample in enumerate(dataset):
        names = _coerce_feature_names(_metadata_value(sample, 'feature_names'))
        if names is None:
            raise ValueError(f'MeshCNN dataset sample {sample_idx} is missing feature_names metadata')
        if names != available_names:
            raise ValueError(f'MeshCNN dataset sample {sample_idx} feature_names differ from sample 0')
        edge_features = getattr(sample, 'edge_features', None)
        if edge_features is not None and int(edge_features.shape[1]) != len(names):
            raise ValueError(
                f'MeshCNN dataset sample {sample_idx} feature_names length {len(names)} '
                f'does not match edge_features dim {int(edge_features.shape[1])}'
            )

    for name in experiment_names:
        missing = [
            feature
            for feature in experiment_feature_selection(name).feature_names
            if feature not in available_names
        ]
        if missing:
            raise ValueError(
                f"MeshCNN dataset is missing requested feature(s) for {name}: {missing}; "
                f"available feature_names={available_names}"
            )


def load_filtered_dataset(path: str, resolution_tag: str) -> list:
    return filter_dataset_by_resolution(load_dataset(path), resolution_tag)


def load_filtered_meshcnn_dataset(path: str, resolution_tag: str) -> list:
    return filter_dataset_by_resolution(load_meshcnn_dataset(path), resolution_tag)


def validate_dataset_roles(args: argparse.Namespace, experiment_names: list[str]) -> dict[str, list]:
    datasets: dict[str, list] = {}
    if is_meshcnn_model(getattr(args, 'model', 'graphsage')):
        if not args.meshcnn_dataset:
            raise ValueError('--meshcnn-dataset is required for sparsemeshcnn')
        datasets['meshcnn'] = load_filtered_meshcnn_dataset(args.meshcnn_dataset, args.resolution_tag)
        validate_meshcnn_dataset_metadata(datasets['meshcnn'], experiment_names)
        return datasets

    if not args.custom_dataset:
        raise ValueError('--custom-dataset is required')
    datasets['custom'] = load_filtered_dataset(args.custom_dataset, args.resolution_tag)
    validate_custom_dataset_metadata(datasets['custom'], experiment_names)
    return datasets


def _validate_split_metadata(payload: dict[str, Any], split_json: Path, args: argparse.Namespace, seed: int) -> None:
    missing = [
        key for key in ('train_group_ids', 'val_group_ids', 'test_group_ids', 'seed', 'resolution_tag')
        if key not in payload
    ]
    if missing:
        raise ValueError(f"{split_json} missing required field(s): {', '.join(sorted(missing))}")
    if int(payload['seed']) != int(seed):
        raise ValueError(f"{split_json} seed={payload['seed']!r} does not match requested seed={seed}")
    if payload.get('resolution_tag') != args.resolution_tag:
        raise ValueError(
            f"{split_json} resolution_tag={payload.get('resolution_tag')!r} "
            f"does not match requested resolution_tag={args.resolution_tag!r}"
        )
    if payload.get('dataset_path') not in (None, ''):
        raise ValueError(
            f"{split_json} is tied to dataset_path={payload.get('dataset_path')!r}; "
            "ablation splits must be dataset-agnostic so paper14 and custom runs reuse identical groups"
        )


def generate_split_files(
    *,
    source_dataset: list,
    splits_dir: Path,
    seeds: list[int],
    resolution_tag: str,
    val_ratio: float,
    test_ratio: float,
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        split_path = split_path_for_seed(splits_dir, seed)
        if split_path.exists():
            continue
        split_dataset(
            source_dataset,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_json_out=split_path,
            dataset_path=None,
            resolution_tag=resolution_tag,
        )


def validate_split_files(args: argparse.Namespace, datasets: dict[str, list]) -> None:
    for seed in args.seeds:
        split_json = split_path_for_seed(Path(args.splits_dir), seed)
        if not split_json.exists():
            raise ValueError(f"missing split JSON for seed {seed}: {split_json}")
        payload = load_split_json_metadata(split_json)
        _validate_split_metadata(payload, split_json, args, seed)
        for role, dataset in datasets.items():
            split_dataset(
                dataset,
                split_json_in=split_json,
                dataset_path=None,
                resolution_tag=args.resolution_tag,
            )


def build_train_command(
    *,
    spec: ExperimentSpec,
    dataset: str | None,
    meshcnn_dataset: str | None = None,
    run_dir: Path,
    split_json: Path,
    seed: int,
    resolution_tag: str,
    epochs: int,
    model: str = 'graphsage',
) -> list[str]:
    if model not in ABLATION_MODELS:
        choices = ', '.join(ABLATION_MODELS)
        raise ValueError(f"unsupported ablation model {model!r}; choose one of: {choices}")
    if is_meshcnn_model(model):
        if not meshcnn_dataset:
            raise ValueError(f'{spec.name} requires a MeshCNN dataset')
        command = [
            sys.executable,
            str(SPARSE_MESHCNN_TRAIN_SCRIPT),
            '--dataset',
            meshcnn_dataset,
            '--run-dir',
            str(run_dir),
            '--epochs',
            str(epochs),
            '--seed',
            str(seed),
            '--split-json-in',
            str(split_json),
            '--resolution-tag',
            resolution_tag,
            '--feature-group',
            spec.feature_group,
        ]
        if spec.enable_ao:
            command.append('--enable-ao')
        if spec.enable_dihedral:
            command.append('--enable-dihedral')
        if spec.enable_symmetry:
            command.append('--enable-symmetry')
        if spec.enable_density:
            command.append('--enable-density')
        if spec.enable_thickness_sdf:
            command.append('--enable-thickness-sdf')
        return command

    if not dataset:
        raise ValueError(f'{spec.name} requires a dataset')

    command = [
        sys.executable,
        str(Path('tools') / 'run_baseline.py'),
        '--model',
        model,
        '--dataset',
        dataset,
        '--run-dir',
        str(run_dir),
        '--resolution-tag',
        resolution_tag,
        '--seed',
        str(seed),
        '--split-json-in',
        str(split_json),
        '--epochs',
        str(epochs),
        '--feature-group',
        spec.feature_group,
    ]
    if model == 'graphsage':
        command += ['--preset', 'paper']
    if spec.enable_ao:
        command.append('--enable-ao')
    if spec.enable_dihedral:
        command.append('--enable-dihedral')
    if spec.enable_symmetry:
        command.append('--enable-symmetry')
    if spec.enable_density:
        command.append('--enable-density')
    if spec.enable_thickness_sdf:
        command.append('--enable-thickness-sdf')
    return command


def collect_success_record(seed: int, run_dir: Path, split_json: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / 'summary.json')
    metrics_05 = summary.get('test_metrics_threshold_0_5', {})
    metrics_best = summary.get('test_metrics_best_validation_threshold', {})

    record: dict[str, Any] = {
        'seed': seed,
        'status': 'completed',
        'run_dir': str(run_dir),
        'split_json': str(split_json),
        'best_epoch': summary.get('best_epoch'),
        'best_val_threshold': summary.get('best_validation_threshold'),
        'resolution_selector': summary.get('resolution_selector', summary.get('resolution_tag')),
        'filtered_graph_count': summary.get('filtered_graph_count'),
    }
    for metric in METRIC_KEYS:
        record[f'{THRESHOLD_05_PREFIX}_{metric}'] = metrics_05.get(metric)
        record[f'{VAL_BEST_PREFIX}_{metric}'] = metrics_best.get(metric)
    return record


def failure_record(seed: int, run_dir: Path, split_json: Path, error: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        'seed': seed,
        'status': 'failed',
        'run_dir': str(run_dir),
        'split_json': str(split_json),
        'best_epoch': None,
        'best_val_threshold': None,
        'resolution_selector': None,
        'filtered_graph_count': None,
        'error': error,
    }
    for column in _metric_columns(THRESHOLD_05_PREFIX) + _metric_columns(VAL_BEST_PREFIX):
        record[column] = None
    return record


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    aggregates: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for prefix in (THRESHOLD_05_PREFIX, VAL_BEST_PREFIX):
        aggregates[prefix] = {}
        for metric in METRIC_KEYS:
            column = f'{prefix}_{metric}'
            values = [
                float(record[column])
                for record in records
                if record.get('status') == 'completed' and record.get(column) is not None
            ]
            aggregates[prefix][metric] = {
                'mean': statistics.mean(values) if values else None,
                'std': statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
                'n': len(values),
            }
    return aggregates


def paired_delta_summary(
    *,
    experiment_name: str,
    experiment_records: list[dict[str, Any]],
    control_name: str,
    control_records: list[dict[str, Any]],
) -> dict[str, Any]:
    experiment_by_seed = {
        int(record['seed']): record for record in experiment_records if record.get('status') == 'completed'
    }
    control_by_seed = {
        int(record['seed']): record for record in control_records if record.get('status') == 'completed'
    }
    paired_seeds = sorted(set(experiment_by_seed) & set(control_by_seed))

    rows: list[dict[str, Any]] = []
    for seed in paired_seeds:
        row: dict[str, Any] = {'seed': seed}
        experiment_record = experiment_by_seed[seed]
        control_record = control_by_seed[seed]
        for prefix in (VAL_BEST_PREFIX, THRESHOLD_05_PREFIX):
            for metric in DELTA_METRIC_KEYS:
                column = f'{prefix}_{metric}'
                row[f'delta_{column}'] = (
                    float(experiment_record[column]) - float(control_record[column])
                    if experiment_record.get(column) is not None and control_record.get(column) is not None
                    else None
                )
        rows.append(row)

    def summarize(prefix: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for metric in DELTA_METRIC_KEYS:
            column = f'delta_{prefix}_{metric}'
            values = [float(row[column]) for row in rows if row.get(column) is not None]
            payload[column] = {
                'mean': statistics.mean(values) if values else None,
                'std': statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
                'n': len(values),
            }
        payload['win_count_fpr'] = sum(
            1 for row in rows if row.get(f'delta_{prefix}_fpr') is not None and row[f'delta_{prefix}_fpr'] < 0
        )
        payload['win_count_f1'] = sum(
            1 for row in rows if row.get(f'delta_{prefix}_f1') is not None and row[f'delta_{prefix}_f1'] > 0
        )
        return payload

    return {
        'experiment': experiment_name,
        'control': control_name,
        'paired_seed_count': len(rows),
        'paired_seeds': paired_seeds,
        'per_seed': rows,
        'val_best': summarize(VAL_BEST_PREFIX),
        'threshold_0_5_diagnostics': summarize(THRESHOLD_05_PREFIX),
    }


def build_experiment_payload(
    *,
    args: argparse.Namespace,
    spec: ExperimentSpec,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    selection = experiment_feature_selection(spec.name)
    model = getattr(args, 'model', 'graphsage')
    dataset = args.meshcnn_dataset if is_meshcnn_model(model) else args.custom_dataset
    return {
        'experiment': spec.name,
        'model': model,
        'dataset': dataset,
        'feature_group': spec.feature_group,
        'feature_flags': selection.feature_flags.as_dict(),
        'feature_names': list(selection.feature_names),
        'resolution_tag': args.resolution_tag,
        'epochs': args.epochs,
        'seeds': args.seeds,
        'splits_dir': str(args.splits_dir),
        'runs': records,
        'aggregates': aggregate_records(records),
    }


def write_experiment_reports(experiment_dir: Path, payload: dict[str, Any]) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_dir / 'summary.json', payload)
    _write_json(experiment_dir / 'per_seed_summary.json', {'runs': payload['runs']})
    _write_json(experiment_dir / 'aggregate_summary.json', payload['aggregates'])

    columns = [
        'seed',
        'status',
        'run_dir',
        'split_json',
        'best_epoch',
        'best_val_threshold',
        'resolution_selector',
        'filtered_graph_count',
        *_metric_columns(THRESHOLD_05_PREFIX),
        *_metric_columns(VAL_BEST_PREFIX),
        'error',
    ]
    with (experiment_dir / 'per_seed_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in payload['runs']:
            writer.writerow(record)

    with (experiment_dir / 'aggregate_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['prefix', 'metric', 'mean', 'std', 'n'])
        writer.writeheader()
        for prefix in (VAL_BEST_PREFIX, THRESHOLD_05_PREFIX):
            for metric in METRIC_KEYS:
                row = payload['aggregates'][prefix][metric]
                writer.writerow({'prefix': prefix, 'metric': metric, **row})


def write_delta_reports(path_prefix: Path, payload: dict[str, Any]) -> None:
    _write_json(path_prefix.with_suffix('.json'), payload)
    columns = ['seed', *_delta_columns(VAL_BEST_PREFIX), *_delta_columns(THRESHOLD_05_PREFIX)]
    with path_prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in payload['per_seed']:
            writer.writerow(row)


def print_experiment_result(name: str, records: list[dict[str, Any]]) -> None:
    completed = [record for record in records if record.get('status') == 'completed']
    failed = len(records) - len(completed)
    if not completed:
        print(f"{name}: 0/{len(records)} completed")
        return
    f1_values = [float(record[f'{VAL_BEST_PREFIX}_f1']) for record in completed]
    fpr_values = [float(record[f'{VAL_BEST_PREFIX}_fpr']) for record in completed]
    print(
        f"{name}: {len(completed)}/{len(records)} completed, "
        f"F1 {statistics.mean(f1_values):.4f}, FPR {statistics.mean(fpr_values):.4f}"
        + (f", failed {failed}" if failed else '')
    )


def run_experiment(
    *,
    args: argparse.Namespace,
    spec: ExperimentSpec,
    runner=subprocess.run,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model = getattr(args, 'model', 'graphsage')
    experiment_dir = Path(args.output_root) / model / 'experiments' / spec.name
    for seed in args.seeds:
        split_json = split_path_for_seed(Path(args.splits_dir), seed)
        run_dir = experiment_dir / f'seed_{seed}'
        run_dir.mkdir(parents=True, exist_ok=True)
        command = build_train_command(
            spec=spec,
            dataset=args.custom_dataset,
            meshcnn_dataset=getattr(args, 'meshcnn_dataset', None),
            run_dir=run_dir,
            split_json=split_json,
            seed=seed,
            resolution_tag=args.resolution_tag,
            epochs=args.epochs,
            model=model,
        )

        print(f"{spec.name} seed {seed}: running")
        try:
            runner(command, check=True)
            records.append(collect_success_record(seed, run_dir, split_json))
        except subprocess.CalledProcessError as exc:
            records.append(failure_record(seed, run_dir, split_json, f'train runner exited with {exc.returncode}'))
            if not args.keep_going:
                return records
        except Exception as exc:
            records.append(failure_record(seed, run_dir, split_json, str(exc)))
            if not args.keep_going:
                return records
    return records


def run_suite(args: argparse.Namespace, runner=subprocess.run) -> dict[str, dict[str, Any]]:
    experiment_names = list(args.experiments)
    validate_experiment_selection(experiment_names, args.model)
    datasets = validate_dataset_roles(args, experiment_names)
    splits_dir = Path(args.splits_dir)

    if args.generate_splits:
        source_dataset = datasets.get('custom') or datasets.get('meshcnn')
        if source_dataset is None:
            raise ValueError('no dataset available for split generation')
        generate_split_files(
            source_dataset=source_dataset,
            splits_dir=splits_dir,
            seeds=args.seeds,
            resolution_tag=args.resolution_tag,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    validate_split_files(args, datasets)
    if args.only_generate_splits:
        print(f"split files ready -> {splits_dir}")
        return {}

    output_root = Path(args.output_root) / args.model
    output_root.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {}
    for name in experiment_names:
        spec = get_experiment_spec(name)
        records = run_experiment(args=args, spec=spec, runner=runner)
        payload = build_experiment_payload(args=args, spec=spec, records=records)
        payloads[name] = payload
        write_experiment_reports(output_root / 'experiments' / name, payload)
        print_experiment_result(name, records)
        if any(record['status'] == 'failed' for record in records) and not args.keep_going:
            break

    write_suite_reports(output_root, payloads)
    return payloads


def write_suite_reports(output_root: Path, payloads: dict[str, dict[str, Any]]) -> None:
    _write_json(output_root / 'suite_summary.json', {'experiments': payloads})

    if 'control14' in payloads:
        control_records = payloads['control14']['runs']
        suite_deltas: dict[str, Any] = {}
        for name, payload in payloads.items():
            delta = paired_delta_summary(
                experiment_name=name,
                experiment_records=payload['runs'],
                control_name='control14',
                control_records=control_records,
            )
            suite_deltas[name] = delta
            write_delta_reports(
                output_root / 'experiments' / name / 'paired_delta_vs_control14',
                delta,
            )
        _write_json(output_root / 'paired_deltas_vs_control14.json', suite_deltas)


def parser_epilog() -> str:
    return """Examples:
  python preprocessing/build_gnn_dataset.py <mesh_dir> --feature-group custom --endpoint-order random --enable-ao --enable-dihedral --enable-symmetry --enable-density --enable-thickness-sdf --save --overwrite --output <custom_dataset.pt>
  python tools/run_feature_ablations.py --model graphsage --custom-dataset <custom_dataset.pt> --experiments control14 ao_density ao_dihedral_symmetry ao_dihedral_symmetry_density_sdf --seeds 7 11 19 --epochs 100 --output-root <out_dir> --generate-splits
  python tools/run_feature_ablations.py --model gatv2 --custom-dataset <custom_dataset.pt> --full-suite --seeds 7 11 19 --epochs 100 --output-root <out_dir> --generate-splits
  python tools/run_feature_ablations.py --model sparsemeshcnn --meshcnn-dataset <meshcnn_superset.pt> --full-suite --seeds 7 11 19 --epochs 100 --output-root <out_dir> --generate-splits
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run fixed-split baseline feature ablations with endpoint-order safety checks.',
        epilog=parser_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', choices=ABLATION_MODELS, default='graphsage')
    parser.add_argument('--custom-dataset', default=None, help='custom superset dual dataset')
    parser.add_argument('--meshcnn-dataset', default=None, help='MeshCNN custom superset dataset')
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=tuple(EXPERIMENT_SPECS),
        default=['control14'],
    )
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--resolution-tag', default='all')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--splits-dir', default=None)
    parser.add_argument('--generate-splits', action='store_true', help='create missing seed split JSONs before runs')
    parser.add_argument('--only-generate-splits', action='store_true', help='prepare splits and exit without training')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--keep-going', action='store_true', help='continue after failed seed runs')
    parser.add_argument(
        '--full-suite',
        action='store_true',
        help='run control14 and all feature-combination ablations',
    )
    args = parser.parse_args(argv)
    if args.full_suite:
        args.experiments = list(FULL_ABLATION_SUITE)
    if args.splits_dir is None:
        args.splits_dir = str(Path(args.output_root) / 'splits')
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        payloads = run_suite(args)
    except ValueError as exc:
        raise SystemExit(f'error: {exc}') from exc

    if any(
        record.get('status') == 'failed'
        for payload in payloads.values()
        for record in payload.get('runs', [])
    ):
        raise SystemExit(1)
    if not args.only_generate_splits:
        print(f"reports saved -> {Path(args.output_root) / args.model}")


if __name__ == '__main__':
    main()
