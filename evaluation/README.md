# Evaluation

Maintained evaluation entrypoints:

- `tools/predict_seams.py` for inference
- `tools/evaluate_dir_topology.py` for topology and post-processing evaluation
- `tools/evaluate_saved_models.py` for reevaluating saved checkpoints

## Inference

`tools/predict_seams.py` is the maintained inference bridge.

```bash
.venv/Scripts/python.exe tools/predict_seams.py --mesh-path data/objs/example.obj --model-weights runs/models/graphsage_paper14/best_model.pth --feature-bundle paper14 --output-json outputs/predictions/example.json
```

## Topology Evaluation

`tools/evaluate_dir_topology.py` is the maintained topology and post-processing evaluation entrypoint.

```bash
.venv/Scripts/python.exe tools/evaluate_dir_topology.py --input-dir data/objs --model-weights runs/models/graphsage_paper14/best_model.pth --feature-bundle paper14 --csv-out outputs/predictions/topology.csv
```

## Saved Checkpoint Reevaluation

`tools/evaluate_saved_models.py` is the maintained entrypoint for reevaluating saved checkpoints.

```bash
.venv/Scripts/python.exe tools/evaluate_saved_models.py --runs-root runs/ablations/graphsage --splits-dir runs/ablations/graphsage/splits --custom-dataset datasets/gnn_custom.pt
```

## Deprecated Script

`evaluation/run_evaluation.py` is deprecated/internal. Do not use it as the current workflow entrypoint.
