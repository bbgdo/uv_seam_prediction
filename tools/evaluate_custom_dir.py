from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
import traceback
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# NOTE: We import predict_seams as a MODULE (not specific symbols) so
# that test code can patch attributes on the module cleanly.
import tools.predict_seams as predict_seams  # noqa: E402


@dataclass(frozen=False)
class MeshAblationRow:
    mesh_name: str
    mesh_path: str
    vertex_count: int
    edge_count: int
    v1_status: str
    v1_error: str | None
    v1_seam_count: int
    v1_time_s: float
    v2_status: str
    v2_error: str | None
    v2_seam_count: int
    v2_time_s: float
    jaccard: float
    v1_only_count: int
    v2_only_count: int
    v2_skeleton_removals: int | None
    v2_steiner_calls: int | None
    v2_steiner_edges_added: int | None
    v2_branches_pruned: int | None
    v2_pruning_iterations: int | None
    v2_thick_band_edges_after: int | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Bulk v1-vs-v2 postprocess ablation over a directory of .obj meshes.'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        type=Path,
        help='Directory containing .obj files to evaluate.',
    )
    parser.add_argument(
        '--model-weights',
        required=True,
        type=Path,
        help='Path to .pt model weights (config.json and summary.json expected alongside).',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Optional override; defaults to summary.json best_validation_threshold.',
    )
    parser.add_argument(
        '--feature-bundle',
        default='auto',
        choices=predict_seams.FEATURE_BUNDLES,
    )
    parser.add_argument(
        '--model-type',
        default='auto',
        choices=predict_seams.MODEL_TYPES,
    )
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument(
        '--config-json',
        default=None,
        type=Path,
        help='Override config JSON path; defaults to weights_dir/config.json.',
    )
    parser.add_argument(
        '--summary-json',
        default=None,
        type=Path,
        help='Override summary JSON path; defaults to weights_dir/summary.json.',
    )
    parser.add_argument(
        '--csv-out',
        default=None,
        type=Path,
        help='If provided, write per-mesh metrics to this CSV file.',
    )
    parser.add_argument(
        '--keep-json',
        action='store_true',
        help='If set, write per-mesh v1 and v2 output JSONs into --output-dir for forensic inspection.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        type=Path,
        help='Directory for per-mesh JSONs when --keep-json is set. Required if --keep-json is set.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='If provided, evaluate only the first N meshes (sorted by name).',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-mesh progress logging; only print final report.',
    )
    return parser.parse_args(argv)


def build_base_args(script_args: argparse.Namespace) -> argparse.Namespace:
    """
    Construct a base argparse Namespace suitable for predict_seams.run_prediction.

    Strategy: invoke predict_seams.parse_args with a synthesized minimal argv
    that includes the required flags. This guarantees we get every default
    value that the CLI tool would set, including all the v1 and v2
    postprocess flags. We then override the cross-cutting fields.

    The returned Namespace has placeholder mesh_path and output_json that
    MUST be overridden per mesh.
    """
    argv = [
        '--mesh-path',
        '/__placeholder__.obj',
        '--model-weights',
        str(script_args.model_weights),
        '--output-json',
        '/__placeholder__.json',
        '--feature-bundle',
        script_args.feature_bundle,
        '--model-type',
        script_args.model_type,
        '--device',
        script_args.device,
    ]
    if script_args.threshold is not None:
        argv += ['--threshold', str(script_args.threshold)]
    if script_args.config_json is not None:
        argv += ['--config-json', str(script_args.config_json)]
    if script_args.summary_json is not None:
        argv += ['--summary-json', str(script_args.summary_json)]

    base = predict_seams.parse_args(argv)
    # Force --no-write-all-edges to keep payloads small (we only need
    # seam_edge_indices for metrics).
    base.write_all_edges = False
    return base


def _run_one(
    base_args: argparse.Namespace,
    mesh_path: Path,
    output_json: Path,
    version: str,
) -> tuple[dict[str, Any], float]:
    """
    Run predict_seams.run_prediction once with the given postprocess version.
    Returns (payload_dict, elapsed_seconds).

    Raises whatever predict_seams.run_prediction raises. Caller is
    responsible for catching.
    """
    args = copy(base_args)
    args.mesh_path = str(mesh_path)
    args.output_json = str(output_json)
    args.postprocess_version = version
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        payload = predict_seams.run_prediction(args)
    elapsed = time.perf_counter() - t0
    return payload, elapsed


def _seam_indices_to_set(payload: dict[str, Any]) -> set[int]:
    indices = payload.get('seam_edge_indices', [])
    return set(int(i) for i in indices)


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _v2_telemetry_fields(payload: dict[str, Any]) -> dict[str, int | None]:
    """
    Extract the v2-specific counters from the output payload's diagnostics
    block. Returns a dict with values, or all-None if v2 didn't run.
    """
    d = payload.get('diagnostics') or {}
    pv2 = d.get('postprocess_v2')
    if not pv2:
        return {
            'skeleton_removals': None,
            'steiner_calls': None,
            'steiner_edges_added': None,
            'branches_pruned': None,
            'pruning_iterations': None,
        }
    skel = pv2.get('skeleton', {})
    brid = pv2.get('bridging', {})
    prun = pv2.get('pruning', {})
    return {
        'skeleton_removals': skel.get('removals_committed'),
        'steiner_calls': brid.get('steiner_calls'),
        'steiner_edges_added': brid.get('steiner_edges_added_total'),
        'branches_pruned': prun.get('total_branches_pruned'),
        'pruning_iterations': prun.get('total_iterations'),
    }


def _topology_count(payload: dict[str, Any], key: str) -> int | None:
    """
    Extract topology counters from diagnostics.seam_topology (BEFORE-pipeline
    mask diagnostics; computed once per run regardless of v1/v2).
    Used to surface 'thick_band_edge_count' style metrics.
    Returns None if unavailable.
    """
    d = payload.get('diagnostics') or {}
    st = d.get('seam_topology')
    if not st:
        return None
    return st.get(key)


def evaluate_one_mesh(
    base_args: argparse.Namespace,
    mesh_path: Path,
    keep_json_dir: Path | None,
    tmp_dir: Path,
) -> MeshAblationRow:
    """
    Evaluate one mesh under v1 and v2. Catches all exceptions per-version
    and returns a populated MeshAblationRow.
    """
    mesh_name = mesh_path.name
    mesh_path_str = str(mesh_path.resolve())

    def _output_path_for(version: str) -> Path:
        if keep_json_dir is not None:
            return keep_json_dir / f'{mesh_path.stem}_{version}.json'
        return tmp_dir / f'{mesh_path.stem}_{version}.json'

    v1_payload: dict[str, Any] | None = None
    v1_status = 'failed'
    v1_error: str | None = None
    v1_time = -1.0
    try:
        v1_payload, v1_time = _run_one(base_args, mesh_path, _output_path_for('v1'), 'v1')
        v1_status = 'ok'
    except Exception as exc:
        v1_error = f'{type(exc).__name__}: {exc}'

    v2_payload: dict[str, Any] | None = None
    v2_status = 'failed'
    v2_error: str | None = None
    v2_time = -1.0
    try:
        v2_payload, v2_time = _run_one(base_args, mesh_path, _output_path_for('v2'), 'v2')
        v2_status = 'ok'
    except Exception as exc:
        v2_error = f'{type(exc).__name__}: {exc}'

    if keep_json_dir is not None:
        if v1_payload is not None:
            predict_seams.write_json_payload(_output_path_for('v1'), v1_payload)
        if v2_payload is not None:
            predict_seams.write_json_payload(_output_path_for('v2'), v2_payload)

    vertex_count = 0
    edge_count = 0
    for payload in (v1_payload, v2_payload):
        if payload is not None:
            topo = payload.get('topology', {})
            vertex_count = int(topo.get('vertex_count', 0))
            edge_count = int(topo.get('edge_count', 0))
            break

    v1_set = _seam_indices_to_set(v1_payload) if v1_payload else set()
    v2_set = _seam_indices_to_set(v2_payload) if v2_payload else set()
    v1_count = len(v1_set) if v1_payload else -1
    v2_count = len(v2_set) if v2_payload else -1

    jaccard = -1.0
    v1_only = -1
    v2_only = -1
    if v1_payload is not None and v2_payload is not None:
        jaccard = _jaccard(v1_set, v2_set)
        v1_only = len(v1_set - v2_set)
        v2_only = len(v2_set - v1_set)

    v2_tele = (
        _v2_telemetry_fields(v2_payload)
        if v2_payload
        else {
            'skeleton_removals': None,
            'steiner_calls': None,
            'steiner_edges_added': None,
            'branches_pruned': None,
            'pruning_iterations': None,
        }
    )
    v2_thick_after = _topology_count(v2_payload, 'thick_band_edge_count') if v2_payload else None

    return MeshAblationRow(
        mesh_name=mesh_name,
        mesh_path=mesh_path_str,
        vertex_count=vertex_count,
        edge_count=edge_count,
        v1_status=v1_status,
        v1_error=v1_error,
        v1_seam_count=v1_count,
        v1_time_s=v1_time,
        v2_status=v2_status,
        v2_error=v2_error,
        v2_seam_count=v2_count,
        v2_time_s=v2_time,
        jaccard=jaccard,
        v1_only_count=v1_only,
        v2_only_count=v2_only,
        v2_skeleton_removals=v2_tele['skeleton_removals'],
        v2_steiner_calls=v2_tele['steiner_calls'],
        v2_steiner_edges_added=v2_tele['steiner_edges_added'],
        v2_branches_pruned=v2_tele['branches_pruned'],
        v2_pruning_iterations=v2_tele['pruning_iterations'],
        v2_thick_band_edges_after=v2_thick_after,
    )


def discover_meshes(input_dir: Path, limit: int | None) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f'input directory not found: {input_dir}')
    meshes = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == '.obj')
    if limit is not None:
        meshes = meshes[:limit]
    return meshes


def format_markdown_report(rows: list[MeshAblationRow]) -> str:
    """
    Return a markdown table summarizing the ablation. Includes a
    header table (one row per mesh) and a footer with aggregates.
    """
    lines: list[str] = []
    lines.append('# v1 vs v2 postprocess ablation')
    lines.append('')
    lines.append(f'Meshes evaluated: {len(rows)}')
    ok = [r for r in rows if r.v1_status == 'ok' and r.v2_status == 'ok']
    v1_fail = [r for r in rows if r.v1_status != 'ok']
    v2_fail = [r for r in rows if r.v2_status != 'ok']
    lines.append(f'Both succeeded: {len(ok)}')
    lines.append(f'v1 failures:    {len(v1_fail)}')
    lines.append(f'v2 failures:    {len(v2_fail)}')
    lines.append('')

    lines.append('| mesh | edges | v1 seams | v2 seams | v1 time | v2 time | jaccard | v1-only | v2-only | v2 spurs pruned | v2 thick |')
    lines.append('|------|------:|---------:|---------:|--------:|--------:|--------:|--------:|--------:|----------------:|---------:|')
    for r in rows:
        v1s = str(r.v1_seam_count) if r.v1_seam_count >= 0 else 'FAIL'
        v2s = str(r.v2_seam_count) if r.v2_seam_count >= 0 else 'FAIL'
        v1t = f'{r.v1_time_s:.2f}s' if r.v1_time_s >= 0 else '-'
        v2t = f'{r.v2_time_s:.2f}s' if r.v2_time_s >= 0 else '-'
        jac = f'{r.jaccard:.3f}' if r.jaccard >= 0 else '-'
        v1o = str(r.v1_only_count) if r.v1_only_count >= 0 else '-'
        v2o = str(r.v2_only_count) if r.v2_only_count >= 0 else '-'
        spurs = str(r.v2_branches_pruned) if r.v2_branches_pruned is not None else '-'
        thick = str(r.v2_thick_band_edges_after) if r.v2_thick_band_edges_after is not None else '-'
        lines.append(
            f'| {r.mesh_name} | {r.edge_count} | {v1s} | {v2s} | {v1t} | {v2t} | '
            f'{jac} | {v1o} | {v2o} | {spurs} | {thick} |'
        )

    if ok:
        mean_jaccard = sum(r.jaccard for r in ok) / len(ok)
        total_v1_time = sum(r.v1_time_s for r in ok)
        total_v2_time = sum(r.v2_time_s for r in ok)
        total_v1_seams = sum(r.v1_seam_count for r in ok)
        total_v2_seams = sum(r.v2_seam_count for r in ok)
        total_spurs = sum(r.v2_branches_pruned or 0 for r in ok)
        total_steiner = sum(r.v2_steiner_edges_added or 0 for r in ok)
        lines.append('')
        lines.append('## Aggregate (both succeeded)')
        lines.append('')
        lines.append(f'- Mean Jaccard:                 {mean_jaccard:.4f}')
        lines.append(f'- Total v1 time:                {total_v1_time:.2f}s')
        lines.append(f'- Total v2 time:                {total_v2_time:.2f}s')
        lines.append(f'- Total v1 seam edges:          {total_v1_seams}')
        lines.append(f'- Total v2 seam edges:          {total_v2_seams}')
        lines.append(f'- Total spurs pruned by v2:     {total_spurs}')
        lines.append(f'- Total Steiner edges added v2: {total_steiner}')

    if v1_fail or v2_fail:
        lines.append('')
        lines.append('## Failures')
        lines.append('')
        for r in v1_fail:
            lines.append(f'- v1 FAILED on {r.mesh_name}: {r.v1_error}')
        for r in v2_fail:
            lines.append(f'- v2 FAILED on {r.mesh_name}: {r.v2_error}')

    return '\n'.join(lines) + '\n'


def write_csv(rows: list[MeshAblationRow], path: Path) -> None:
    """
    Write the per-mesh rows to CSV. Columns are deterministic; one
    row per mesh including failed runs (with sentinel values).
    """
    import csv

    fieldnames = [
        'mesh_name',
        'mesh_path',
        'vertex_count',
        'edge_count',
        'v1_status',
        'v1_error',
        'v1_seam_count',
        'v1_time_s',
        'v2_status',
        'v2_error',
        'v2_seam_count',
        'v2_time_s',
        'jaccard',
        'v1_only_count',
        'v2_only_count',
        'v2_skeleton_removals',
        'v2_steiner_calls',
        'v2_steiner_edges_added',
        'v2_branches_pruned',
        'v2_pruning_iterations',
        'v2_thick_band_edges_after',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.keep_json and args.output_dir is None:
        print('ERROR: --keep-json requires --output-dir', file=sys.stderr)
        return 2

    try:
        base = build_base_args(args)
    except predict_seams.PredictionError as exc:
        print(f'ERROR: setup failed: {exc}', file=sys.stderr)
        return 3

    try:
        meshes = discover_meshes(args.input_dir, args.limit)
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 4

    if not meshes:
        print(f'ERROR: no .obj files found in {args.input_dir}', file=sys.stderr)
        return 5

    rows: list[MeshAblationRow] = []
    keep_json_dir: Path | None = None
    if args.keep_json:
        keep_json_dir = args.output_dir
        keep_json_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='ablation_') as tmp:
        tmp_path = Path(tmp)
        for i, mesh_path in enumerate(meshes, 1):
            if not args.quiet:
                print(f'[{i}/{len(meshes)}] {mesh_path.name} ...', file=sys.stderr, flush=True)
            row = evaluate_one_mesh(
                base_args=base,
                mesh_path=mesh_path,
                keep_json_dir=keep_json_dir,
                tmp_dir=tmp_path,
            )
            rows.append(row)
            if not args.quiet:
                msg_parts = [
                    f'  v1={row.v1_seam_count}',
                    f'v2={row.v2_seam_count}',
                    f'jac={row.jaccard:.3f}' if row.jaccard >= 0 else 'jac=-',
                ]
                print('  ' + ' '.join(msg_parts), file=sys.stderr, flush=True)

    report = format_markdown_report(rows)
    print(report)

    if args.csv_out is not None:
        write_csv(rows, args.csv_out)
        print(f'CSV written: {args.csv_out}', file=sys.stderr)

    both_ok = sum(1 for r in rows if r.v1_status == 'ok' and r.v2_status == 'ok')
    return 0 if both_ok > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
