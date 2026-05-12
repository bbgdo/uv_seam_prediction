from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from tools.utils.json_io import write_json  # noqa: E402
from tools.utils.reeval_common import AGGREGATE_FILENAME, REEVAL_FILENAME  # noqa: E402
from tools.utils.reeval_reporting import aggregate_reevaluations, load_reference_control_reevaluations  # noqa: E402
from tools.utils.reeval_runs import discover_saved_runs, evaluate_saved_run, resolve_device  # noqa: E402
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
    'discover_saved_runs',
    'exact_validation_threshold',
    'load_reference_control_reevaluations',
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Offline reevaluate saved GraphSAGE/GATv2 best_model.pth checkpoints.')
    parser.add_argument('--runs-root', required=True, help='root directory containing experiment outputs')
    parser.add_argument('--splits-dir', required=True, help='directory containing frozen seed split JSON files')
    parser.add_argument('--gnn-dataset', default=None, help='GNN dataset override for saved GraphSAGE/GATv2 runs')
    parser.add_argument('--experiments', nargs='+', default=None, help='experiment names to reevaluate')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='seed numbers to reevaluate')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--report-grid', default=None, help='comma list or start:stop:step threshold grid')
    parser.add_argument('--reference-control-dir', default=None, help='previously reevaluated control experiment dir')
    parser.add_argument('--dry-run', action='store_true', help='show matching runs without running inference')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report_grid = build_report_grid(args.report_grid)
        targets = discover_saved_runs(args)
        if args.dry_run:
            for target in targets:
                print(
                    f"would evaluate: experiment={target.experiment or '-'} seed={target.seed} "
                    f"run_dir={target.run_dir}"
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
            payload = evaluate_saved_run(target, device=device, report_grid=report_grid)
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
    except ValueError as exc:
        raise SystemExit(f'error: {exc}') from exc


if __name__ == '__main__':
    main()
