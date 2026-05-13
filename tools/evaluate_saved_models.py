from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from tools.utils.json_io import write_json  # noqa: E402
from tools.utils.reeval_common import AGGREGATE_FILENAME, REEVAL_FILENAME  # noqa: E402
from tools.utils.reeval_gnn import discover_saved_gnn_runs, evaluate_saved_gnn_run, resolve_device  # noqa: E402
from tools.utils.reeval_materialize import materialize_reeval_reports  # noqa: E402
from tools.utils.reeval_meshcnn import discover_saved_meshcnn_runs, evaluate_saved_meshcnn_run  # noqa: E402
from tools.utils.reeval_reporting import aggregate_reevaluations, load_reference_control_reevaluations  # noqa: E402
from tools.utils.reeval_thresholds import (  # noqa: E402
    build_report_grid,
    compute_threshold_metrics_fast,
    exact_validation_threshold,
)


__all__ = [
    'AGGREGATE_FILENAME',
    'REEVAL_FILENAME',
    'aggregate_reevaluations',
    'build_report_grid',
    'compute_threshold_metrics_fast',
    'discover_saved_gnn_runs',
    'discover_saved_meshcnn_runs',
    'evaluate_saved_gnn_run',
    'evaluate_saved_meshcnn_run',
    'exact_validation_threshold',
    'load_reference_control_reevaluations',
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Offline reevaluate saved GraphSAGE, GATv2, and SparseMeshCNN checkpoints.')
    parser.add_argument(
        '--model',
        choices=['auto', 'graphsage', 'gatv2', 'sparsemeshcnn'],
        default='auto',
        help='model family to reevaluate; auto infers from saved configs',
    )
    parser.add_argument('--runs-root', required=True, help='root directory containing experiment outputs')
    parser.add_argument('--splits-dir', default=None, help='directory containing frozen seed split JSON files')
    parser.add_argument('--gnn-dataset', default=None, help='GNN dataset override for saved GraphSAGE/GATv2 runs')
    parser.add_argument('--meshcnn-dataset', default=None, help='MeshCNN dataset override for saved SparseMeshCNN runs')
    parser.add_argument('--experiments', nargs='+', default=None, help='experiment names to reevaluate')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='seed numbers to reevaluate')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--report-grid', default=None, help='comma list or start:stop:step threshold grid')
    parser.add_argument(
        '--threshold-decimals',
        type=int,
        default=3,
        help='search thresholds on a fixed decimal grid; use -1 for exact score breakpoints',
    )
    parser.add_argument('--reference-control-dir', default=None, help='previously reevaluated control experiment dir')
    parser.add_argument('--materialize-reports', action='store_true', help='update ablation suite reports from reevaluation files')
    parser.add_argument('--materialize-only', action='store_true', help='only update reports from existing reevaluation files')
    parser.add_argument(
        '--update-run-summaries',
        action='store_true',
        help='with materialization, update each seed summary.json threshold and test metrics',
    )
    parser.add_argument('--dry-run', action='store_true', help='show matching runs without running inference')
    args = parser.parse_args(argv)
    if args.threshold_decimals < -1:
        parser.error('--threshold-decimals must be -1 or non-negative')
    if args.materialize_only:
        args.materialize_reports = True
    if not args.materialize_only and not args.splits_dir:
        parser.error('--splits-dir is required unless --materialize-only is used')
    return args


def infer_model(args: argparse.Namespace) -> str:
    if args.model != 'auto':
        return args.model
    for config_path in sorted(Path(args.runs_root).rglob('config.json')):
        try:
            with config_path.open(encoding='utf-8') as handle:
                config = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(config, dict):
            if config.get('model') == 'sparsemeshcnn':
                return 'sparsemeshcnn'
            model_name = config.get('model_name')
            if model_name in {'graphsage', 'gatv2'}:
                return str(model_name)
    root_name = Path(args.runs_root).name.lower()
    if 'meshcnn' in root_name:
        return 'sparsemeshcnn'
    return 'graphsage'


def threshold_decimals_arg(args: argparse.Namespace) -> int | None:
    return None if args.threshold_decimals == -1 else int(args.threshold_decimals)


def discover_targets(args: argparse.Namespace):
    model = infer_model(args)
    if model == 'sparsemeshcnn':
        return discover_saved_meshcnn_runs(args)
    return discover_saved_gnn_runs(args)


def evaluate_target(target, *, args: argparse.Namespace, device, report_grid: list[float]) -> dict:
    threshold_decimals = threshold_decimals_arg(args)
    model = infer_model(args)
    if model == 'sparsemeshcnn':
        return evaluate_saved_meshcnn_run(
            target,
            device=device,
            report_grid=report_grid,
            threshold_decimals=threshold_decimals,
        )
    return evaluate_saved_gnn_run(
        target,
        device=device,
        report_grid=report_grid,
        threshold_decimals=threshold_decimals,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report_grid = build_report_grid(args.report_grid)
        if args.materialize_only:
            materialize_reeval_reports(args)
            return

        targets = discover_targets(args)
        if args.dry_run:
            for target in targets:
                print(
                    f"would evaluate: experiment={target.experiment or '-'} seed={target.seed} "
                    f"model={infer_model(args)} run_dir={target.run_dir}"
                )
            return

        reference_control = (
            load_reference_control_reevaluations(args.reference_control_dir)
            if args.reference_control_dir
            else None
        )
        device = resolve_device(args.device)
        results = []
        for target in targets:
            print(f"reevaluating {target.experiment or 'run'} seed {target.seed}: {target.run_dir}")
            payload = evaluate_target(target, args=args, device=device, report_grid=report_grid)
            results.append(payload)
            write_json(target.run_dir / REEVAL_FILENAME, payload)
            print(f"  wrote {target.run_dir / REEVAL_FILENAME}")

        if len(results) > 1 or reference_control is not None:
            aggregate = aggregate_reevaluations(results, reference_control=reference_control)
            output_path = Path(args.runs_root) / AGGREGATE_FILENAME
            write_json(output_path, aggregate)
            print(f"aggregate written -> {output_path}")
            for experiment, delta in aggregate.get('paired_delta_vs_reference_control', {}).items():
                skipped = delta.get('skipped_seeds') or []
                if skipped:
                    print(
                        f"warning: external control pairing for {experiment} skipped {len(skipped)} seed(s)",
                        file=sys.stderr,
                    )
        if args.materialize_reports:
            materialize_reeval_reports(args)
    except ValueError as exc:
        raise SystemExit(f'error: {exc}') from exc


if __name__ == '__main__':
    main()
