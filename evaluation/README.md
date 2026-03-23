# Evaluation — UV Quality Pipeline

End-to-end UV quality evaluation comparing:
- **Ours** — GNN-predicted seams → Blender unwrap
- **Smart UV Project** — Blender heuristic baseline
- **Ground Truth** — artist UV embedded in the original `.obj`

Metrics are adapted from *"A Dataset and Benchmark for Mesh Parameterization"* (arXiv:2208.01772).

---

## Files

| File | Description |
|---|---|
| `uv_metrics.py` | Pure numpy UV quality metrics + `.obj` parser (no Blender needed) |
| `blender_unwrap.py` | Blender script: mark seams → unwrap → export (runs via `blender -b -P ...`) |
| `run_evaluation.py` | Orchestrator: inference → Blender → metrics → JSON + plots |
| `compare_models.py` | Cross-model comparison table and bar plots for the diploma thesis |
| `results/` | Output directory (JSON + PNGs) |

---

## UV Metrics (`uv_metrics.py`)

All metrics work on triangulated meshes with vertex positions and UV coordinates.

| Metric | Description | Better |
|---|---|---|
| `area_distortion_avg` | Area-weighted mean of `A_3d/A_uv + A_uv/A_3d - 2` | lower |
| `area_distortion_max` | Worst-case area distortion across all triangles | lower |
| `angle_distortion_avg` | Area-weighted mean of `σ1/σ2 + σ2/σ1 - 2` (via Jacobian SVD) | lower |
| `angle_distortion_max` | Worst-case angle distortion | lower |
| `symmetric_dirichlet_avg` | Area-weighted mean of `(σ1² + σ2² + 1/σ1² + 1/σ2²) / 2` | lower |
| `flipped_pct` | % of UV triangles with flipped orientation | lower |
| `num_shells` | UV island count (connected components in UV space) | context-dependent |
| `seam_length` | Total 3D seam length / sqrt(total mesh area) | context-dependent |

```bash
python evaluation/uv_metrics.py path/to/mesh.obj
```

---

## Blender Unwrap (`blender_unwrap.py`)

Requires Blender 4.0+ installed and callable as `blender` (or full path).

**Unwrap with predicted seams:**
```bash
blender -b --factory-startup -P evaluation/blender_unwrap.py -- \
    --input mesh.obj \
    --seams seam_indices.txt \
    --output mesh_unwrapped.obj \
    --method ANGLE_BASED
```

**Smart UV Project baseline:**
```bash
blender -b --factory-startup -P evaluation/blender_unwrap.py -- \
    --input mesh.obj \
    --output mesh_smart_uv.obj \
    --smart-uv
```

**Ground truth passthrough (re-export existing UV):**
```bash
blender -b --factory-startup -P evaluation/blender_unwrap.py -- \
    --input mesh.obj \
    --output mesh_gt.obj \
    --preserve-uv
```

---

## Full Evaluation (`run_evaluation.py`)

Runs the entire pipeline for a model on all test meshes.

```bash
python evaluation/run_evaluation.py \
    --test-meshes ./3d-objs/ \
    --dual-dataset dataset_dual.pt \
    --weights runs/dual_graphsage_001/best_model.pth \
    --model-type graphsage \
    --blender-exe blender \
    --output-dir evaluation/results/graphsage_eval \
    --threshold 0.5 \
    --max-meshes 2          # optional: limit for quick testing
```

**Outputs in `--output-dir`:**

| File | Description |
|---|---|
| `per_mesh_results.json` | Per-mesh metrics for GT, predicted, and Smart UV |
| `summary.json` | Aggregated means and standard deviations |
| `comparison_table.png` | Rendered metrics table (best values highlighted) |
| `distortion_bars.png` | Grouped bar chart of distortion metrics |
| `shells_comparison.png` | UV shell count comparison |
| `per_mesh_scatter.png` | Predicted vs GT distortion per mesh |

---

## Cross-Model Comparison (`compare_models.py`)

After running `run_evaluation.py` for each model:

```bash
python evaluation/compare_models.py \
    evaluation/results/graphsage_eval \
    evaluation/results/gatv2_eval \
    --output-dir evaluation/results/comparison
```

**Outputs:**

| File | Description |
|---|---|
| `full_comparison_table.png` | Diploma thesis table: all models × all metrics |
| `method_comparison_bars.png` | Grouped bar chart across models + baselines |
| `comparison_summary.json` | Merged summary data |
