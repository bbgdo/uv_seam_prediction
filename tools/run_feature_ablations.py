from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from tools._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from tools.utils.ablation_datasets import (  # noqa: E402
    get_gnn_dataset_arg,
    load_filtered_dataset,
    load_filtered_meshcnn_dataset,
    validate_dataset_roles_with_loaders,
    validate_gnn_dataset_metadata,
    validate_meshcnn_dataset_metadata,
)
from tools.utils.ablation_reports import (  # noqa: E402
    THRESHOLD_05_PREFIX,
    VAL_BEST_PREFIX,
    build_experiment_payload,
    load_existing_suite_payloads,
    paired_delta_summary,
    print_experiment_result,
    write_experiment_reports,
    write_suite_reports,
)
from tools.utils.ablation_runner import build_train_command, run_experiment  # noqa: E402
from tools.utils.ablation_splits import (  # noqa: E402
    generate_split_files,
    split_json_for_seed,
    split_path_for_seed,
    validate_split_files,
)
from tools.utils.ablation_specs import (  # noqa: E402
    ALL_COMBINATORIAL_SUITE,
    ALL_EXPERIMENT_SPECS,
    ABLATION_MODELS,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    EXPERIMENT_SPECS,
    FULL_ABLATION_SUITE,
    PHASE_ADD_ONE,
    PHASE_BASELINE,
    PHASE_PAIRWISE,
    VALID_COMBINATORIAL_COUNTS,
    combinatorial_suite,
    experiment_feature_selection,
    get_experiment_spec,
    validate_experiment_selection,
)

__all__ = [
    'ALL_COMBINATORIAL_SUITE',
    'ALL_EXPERIMENT_SPECS',
    'EXPERIMENT_SPECS',
    'FULL_ABLATION_SUITE',
    'THRESHOLD_05_PREFIX',
    'VAL_BEST_PREFIX',
    'build_experiment_payload',
    'build_train_command',
    'experiment_feature_selection',
    'generate_split_files',
    'get_gnn_dataset_arg',
    'load_existing_suite_payloads',
    'load_filtered_dataset',
    'load_filtered_meshcnn_dataset',
    'paired_delta_summary',
    'parse_args',
    'run_experiment',
    'run_suite',
    'split_json_for_seed',
    'split_path_for_seed',
    'validate_dataset_roles',
    'validate_experiment_selection',
    'validate_gnn_dataset_metadata',
    'validate_meshcnn_dataset_metadata',
    'validate_split_files',
]


def validate_dataset_roles(args: argparse.Namespace, experiment_names: list[str]) -> dict[str, list]:
    return validate_dataset_roles_with_loaders(
        args,
        experiment_names,
        load_gnn_dataset=load_filtered_dataset,
        load_meshcnn_dataset_fn=load_filtered_meshcnn_dataset,
    )


def run_suite(args: argparse.Namespace, runner=subprocess.run) -> dict[str, dict]:
    experiment_names = list(args.experiments)
    validate_experiment_selection(experiment_names, args.model)
    datasets = validate_dataset_roles(args, experiment_names)
    splits_dir = Path(args.splits_dir)

    if getattr(args, 'split_json_in', None) and len(args.seeds) != 1:
        raise ValueError('--split-json-in can only be used with exactly one seed')
    if getattr(args, 'split_json_in', None) and (args.generate_splits or args.only_generate_splits):
        raise ValueError('--split-json-in cannot be combined with split generation flags')

    if args.generate_splits:
        source_dataset = datasets.get('gnn') or datasets.get('meshcnn')
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
    payloads = load_existing_suite_payloads(output_root)
    print(f"{args.model}: feature ablation suite has {len(experiment_names)} experiments per architecture")
    phases = (PHASE_BASELINE, PHASE_ADD_ONE, PHASE_PAIRWISE)
    for phase in phases:
        phase_names = [name for name in experiment_names if get_experiment_spec(name).phase == phase]
        if not phase_names:
            continue
        print(f"{args.model}: phase {phase} ({len(phase_names)} experiments)")
        for name in phase_names:
            spec = get_experiment_spec(name)
            records = run_experiment(args=args, spec=spec, runner=runner)
            payload = build_experiment_payload(args=args, spec=spec, records=records)
            payloads[name] = payload
            write_experiment_reports(output_root / 'experiments' / name, payload)
            print_experiment_result(name, records)
            if any(record['status'] == 'failed' for record in records) and not args.keep_going:
                write_suite_reports(output_root, payloads)
                return payloads

    extra_names = [name for name in experiment_names if get_experiment_spec(name).phase not in phases]
    if extra_names:
        print(f"{args.model}: phase combinatorial ({len(extra_names)} experiments)")
        for name in extra_names:
            spec = get_experiment_spec(name)
            records = run_experiment(args=args, spec=spec, runner=runner)
            payload = build_experiment_payload(args=args, spec=spec, records=records)
            payloads[name] = payload
            write_experiment_reports(output_root / 'experiments' / name, payload)
            print_experiment_result(name, records)
            if any(record['status'] == 'failed' for record in records) and not args.keep_going:
                write_suite_reports(output_root, payloads)
                return payloads

    write_suite_reports(output_root, payloads)
    return payloads


def parser_epilog() -> str:
    return """Examples:
  python tools/run_feature_ablations.py --model graphsage --gnn-dataset <custom_dataset.pt> --output-root <out_dir> --generate-splits
  python tools/run_feature_ablations.py --model gatv2 --gnn-dataset <custom_dataset.pt> --control14-run-dir <control14_dir> --output-root <out_dir> --generate-splits
  python tools/run_feature_ablations.py --model sparsemeshcnn --meshcnn-dataset <meshcnn_superset.pt> --output-root <out_dir> --generate-splits
  python tools/run_feature_ablations.py --model graphsage --gnn-dataset <custom_dataset.pt> --combinatorial-suite 1 2 3 4 5 --output-root <out_dir> --generate-splits
  python tools/run_feature_ablations.py --model gatv2 --gnn-dataset datasets/600_dual_gnn.pt --control14-run-dir runs/control14 --combinatorial-suite 2 --output-root runs/003_ablations_gatv2 --generate-splits --seeds 11 22 33 --exclude-case ao_dihedral ao_density
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run fixed-split Pairwise Feature Search ablations with endpoint-order safety checks.',
        epilog=parser_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', choices=ABLATION_MODELS, default='graphsage')
    parser.add_argument('--gnn-dataset', default=None, help='GNN custom superset dual dataset')
    parser.add_argument('--meshcnn-dataset', default=None, help='MeshCNN custom superset dataset')
    parser.add_argument('--control14-run-dir', default=None, help='existing control14 experiment/run dir to reuse')
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=tuple(ALL_EXPERIMENT_SPECS),
        default=list(FULL_ABLATION_SUITE),
    )
    parser.add_argument('--seeds', type=int, nargs='+', default=[DEFAULT_SEED], help='training/split seeds')
    parser.add_argument('--resolution-tag', default='all')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--splits-dir', default=None)
    parser.add_argument('--split-json-in', default=None, help='load fixed train/val/test group ids from this JSON file')
    parser.add_argument('--generate-splits', action='store_true', help='create missing seed split JSONs before runs')
    parser.add_argument('--only-generate-splits', action='store_true', help='prepare splits and exit without training')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--keep-going', action='store_true', help='continue after failed seed runs')
    parser.add_argument('--mean_debug', action='store_true', help='use GraphSAGE mean aggregation for debug-speed runs')
    parser.add_argument(
        '--exclude-case',
        dest='exclude_cases',
        nargs='+',
        choices=tuple(ALL_EXPERIMENT_SPECS),
        default=[],
        help='skip named experiment cases while preserving existing reports',
    )
    parser.add_argument(
        '--combinatorial-suite',
        type=int,
        nargs='+',
        choices=VALID_COMBINATORIAL_COUNTS,
        metavar='N',
        default=None,
        help='run baseline plus all custom-feature combinations for each requested feature count',
    )
    args = parser.parse_args(argv)
    if args.combinatorial_suite is not None:
        args.experiments = combinatorial_suite(args.combinatorial_suite)
    if args.exclude_cases:
        excluded = set(args.exclude_cases)
        args.experiments = [name for name in args.experiments if name not in excluded]
    args.seed = args.seeds[0]
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
