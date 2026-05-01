# Evaluation

The maintained evaluation surface is:

- `tools/predict_seams.py` for single-mesh inference
- `tools/evaluate_dir_topology.py` for bulk topology and post-processing evaluation
- `tools/evaluate_saved_models.py` for offline reevaluation of saved checkpoints

## Topology Evaluation

```bash
python tools/evaluate_dir_topology.py \
  --input-dir ./3d-objs \
  --model-weights runs/graphsage_paper14/best_model.pth \
  --feature-bundle paper14 \
  --csv-out topology_eval.csv
```

This utility reuses `tools/predict_seams.py`, records per-mesh seam counts and topology telemetry, and can optionally keep per-mesh JSON payloads for inspection.

## Saved Checkpoint Reevaluation

```bash
python tools/evaluate_saved_models.py \
  --runs-root runs/ablations_graphsage \
  --splits-dir runs/ablations_graphsage/splits \
  --custom-dataset dataset_custom_dual.pt
```

This utility reevaluates stored `best_model.pth` checkpoints, recomputes the exact validation-optimal threshold, and writes per-run and aggregate JSON reports.

## Archived Script

`run_evaluation.py` is kept only for older UV unwrap comparison studies. It is not the maintained evaluation entrypoint for current training and inference flows.
