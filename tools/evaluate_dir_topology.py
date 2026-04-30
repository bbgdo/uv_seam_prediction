from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import time
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the module so tests can patch attributes on it cleanly.
import tools.predict_seams as predict_seams  # noqa: E402


@dataclass(frozen=False)
class MeshTopologyRow:
    mesh_name: str
    mesh_path: str
    vertex_count: int
    edge_count: int
    status: str
    error: str | None
    seam_count: int
    time_s: float
    skeleton_removals: int | None
    bridge_edges_added: int | None
    branches_pruned: int | None
    pruning_iterations: int | None
    thick_band_edges_after: int | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Bulk topology pipeline evaluation over a directory of .obj meshes.'
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
        help='If set, write per-mesh output JSONs into --output-dir for forensic inspection.',
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
    base.write_all_edges = False
    return base


def _run_one(
    base_args: argparse.Namespace,
    mesh_path: Path,
    output_json: Path,
) -> tuple[dict[str, Any], float]:
    args = copy(base_args)
    args.mesh_path = str(mesh_path)
    args.output_json = str(output_json)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        payload = predict_seams.run_prediction(args)
    elapsed = time.perf_counter() - t0
    return payload, elapsed


def _seam_indices_to_set(payload: dict[str, Any]) -> set[int]:
    indices = payload.get('seam_edge_indices', [])
    return set(int(i) for i in indices)


def _telemetry_fields(payload: dict[str, Any]) -> dict[str, int | None]:
    d = payload.get('diagnostics') or {}
    postprocess = d.get('postprocess')
    if not postprocess:
        return {
            'skeleton_removals': None,
            'bridge_edges_added': None,
            'branches_pruned': None,
            'pruning_iterations': None,
        }
    skel = postprocess.get('skeleton', {})
    brid = postprocess.get('bridging', {})
    prun = postprocess.get('pruning', {})
    return {
        'skeleton_removals': skel.get('removals_committed'),
        'bridge_edges_added': brid.get('added_bridge_edges'),
        'branches_pruned': prun.get('total_branches_pruned'),
        'pruning_iterations': prun.get('total_iterations'),
    }


def _topology_count(payload: dict[str, Any], key: str) -> int | None:
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
) -> MeshTopologyRow:
    mesh_name = mesh_path.name
    mesh_path_str = str(mesh_path.resolve())
    output_json = (
        keep_json_dir / f'{mesh_path.stem}.json'
        if keep_json_dir is not None
        else tmp_dir / f'{mesh_path.stem}.json'
    )

    payload: dict[str, Any] | None = None
    status = 'failed'
    error: str | None = None
    elapsed = -1.0
    try:
        payload, elapsed = _run_one(base_args, mesh_path, output_json)
        status = 'ok'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'

    if keep_json_dir is not None and payload is not None:
        predict_seams.write_json_payload(output_json, payload)

    vertex_count = 0
    edge_count = 0
    seam_count = -1
    telemetry = {
        'skeleton_removals': None,
        'bridge_edges_added': None,
        'branches_pruned': None,
        'pruning_iterations': None,
    }
    thick_after = None
    if payload is not None:
        topo = payload.get('topology', {})
        vertex_count = int(topo.get('vertex_count', 0))
        edge_count = int(topo.get('edge_count', 0))
        seam_count = len(_seam_indices_to_set(payload))
        telemetry = _telemetry_fields(payload)
        thick_after = _topology_count(payload, 'thick_band_edge_count')

    return MeshTopologyRow(
        mesh_name=mesh_name,
        mesh_path=mesh_path_str,
        vertex_count=vertex_count,
        edge_count=edge_count,
        status=status,
        error=error,
        seam_count=seam_count,
        time_s=elapsed,
        skeleton_removals=telemetry['skeleton_removals'],
        bridge_edges_added=telemetry['bridge_edges_added'],
        branches_pruned=telemetry['branches_pruned'],
        pruning_iterations=telemetry['pruning_iterations'],
        thick_band_edges_after=thick_after,
    )


def discover_meshes(input_dir: Path, limit: int | None) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f'input directory not found: {input_dir}')
    meshes = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == '.obj')
    if limit is not None:
        meshes = meshes[:limit]
    return meshes


def format_markdown_report(rows: list[MeshTopologyRow]) -> str:
    lines: list[str] = []
    lines.append('# Topology pipeline evaluation')
    lines.append('')
    lines.append(f'Meshes evaluated: {len(rows)}')
    ok = [r for r in rows if r.status == 'ok']
    failed = [r for r in rows if r.status != 'ok']
    lines.append(f'Succeeded: {len(ok)}')
    lines.append(f'Failed:    {len(failed)}')
    lines.append('')

    lines.append('| mesh | edges | seam count | time | skel removals | bridge edges | spurs pruned | thick after |')
    lines.append('|------|------:|-----------:|-----:|--------------:|--------------:|-------------:|------------:|')
    for r in rows:
        seams = str(r.seam_count) if r.seam_count >= 0 else 'FAIL'
        elapsed = f'{r.time_s:.2f}s' if r.time_s >= 0 else '-'
        skel = str(r.skeleton_removals) if r.skeleton_removals is not None else '-'
        bridge = str(r.bridge_edges_added) if r.bridge_edges_added is not None else '-'
        spurs = str(r.branches_pruned) if r.branches_pruned is not None else '-'
        thick = str(r.thick_band_edges_after) if r.thick_band_edges_after is not None else '-'
        lines.append(
            f'| {r.mesh_name} | {r.edge_count} | {seams} | {elapsed} | '
            f'{skel} | {bridge} | {spurs} | {thick} |'
        )

    if ok:
        total_time = sum(r.time_s for r in ok)
        total_seams = sum(r.seam_count for r in ok)
        total_spurs = sum(r.branches_pruned or 0 for r in ok)
        total_bridge = sum(r.bridge_edges_added or 0 for r in ok)
        total_skeleton = sum(r.skeleton_removals or 0 for r in ok)
        lines.append('')
        lines.append('## Aggregate')
        lines.append('')
        lines.append(f'- Total time:                 {total_time:.2f}s')
        lines.append(f'- Total seam edges:           {total_seams}')
        lines.append(f'- Total spurs pruned:         {total_spurs}')
        lines.append(f'- Total bridge edges added:   {total_bridge}')
        lines.append(f'- Total skeleton removals:    {total_skeleton}')

    if failed:
        lines.append('')
        lines.append('## Failures')
        lines.append('')
        for r in failed:
            lines.append(f'- FAILED on {r.mesh_name}: {r.error}')

    return '\n'.join(lines) + '\n'


def write_csv(rows: list[MeshTopologyRow], path: Path) -> None:
    import csv

    fieldnames = [
        'mesh_name',
        'mesh_path',
        'vertex_count',
        'edge_count',
        'status',
        'error',
        'seam_count',
        'time_s',
        'skeleton_removals',
        'bridge_edges_added',
        'branches_pruned',
        'pruning_iterations',
        'thick_band_edges_after',
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

    rows: list[MeshTopologyRow] = []
    keep_json_dir: Path | None = None
    if args.keep_json:
        keep_json_dir = args.output_dir
        keep_json_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='topology_eval_') as tmp:
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
                print(
                    f'  seams={row.seam_count}  spurs={row.branches_pruned}  '
                    f'thick={row.thick_band_edges_after}',
                    file=sys.stderr,
                    flush=True,
                )

    report = format_markdown_report(rows)
    print(report)

    if args.csv_out is not None:
        write_csv(rows, args.csv_out)
        print(f'CSV written: {args.csv_out}', file=sys.stderr)

    return 0 if any(r.status == 'ok' for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
