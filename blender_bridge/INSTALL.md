# UV Seam Predictor — Setup Guide

## Why the DLL error can't be fixed with path tricks

`[WinError 1114] c10.dll initialization failed` is `ERROR_DLL_INIT_FAILED`.
The DLL is *found* — but its `DllMain` returns `FALSE`. This is a **process-level
conflict**: by the time this addon loads, Blender (or one of its other addons —
True-Terrain, botaniq, DECALmachine, etc.) has already initialized OpenMP or another
native library that torch's `c10.dll` also tries to initialize. Windows refuses the
second initialization and there's no fix from Python's side.

**Solution**: run torch in a separate Python process. No shared DLL state, no conflict.

---

## Setup (2 minutes)

### Step 1 — find your Python with torch

This is the Python you used to train the model. Run in a terminal:

```
python -c "import torch, torch_geometric; print(torch.__version__)"
```

If that works, your Python exe is just `python`. If not, use the full path:

```
where python        # Windows — shows all Python executables in PATH
```

Common locations:
- `C:\Users\you\AppData\Local\Programs\Python\Python311\python.exe`
- `C:\Users\you\.venv\Scripts\python.exe` (if you use a venv)
- `C:\ProgramData\miniconda3\envs\myenv\python.exe` (conda)

### Step 2 — configure the addon

1. Enable addon: *Edit → Preferences → Add-ons → Install* → select `blender_bridge/`
   (zip it first, or install `__init__.py` directly).
2. Open the N-panel (**N** key) → **UV Seam GNN**.
3. Set **Python Exe** to the path from Step 1.
4. Click **Test Python** — you should see `torch X.X, torch_geometric X.X` in the
   status bar. If not, fix the path.

### Step 3 — run inference

1. Set **Weights** to `models/graphsage/best_model.pth`.
2. Select a mesh object.
3. Adjust **Threshold** (default 0.5).
4. Click **Auto-Mark UV Seams**.

---

## If torch isn't in your system Python

Install it in a venv or conda env, then point **Python Exe** at that env's Python:

```bash
python -m venv seam_env
seam_env\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

Then set Python Exe to the full path of `seam_env\Scripts\python.exe`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Test Python" fails: `No module named torch` | Wrong Python exe — check with `where python` |
| "Test Python" fails: `python not found` | Use an absolute path to python.exe |
| `model.py not found` | Unrelated to model path — `run_inference.py` embeds the model |
| `Weights file not found` | Use the file picker; verify `.pth` exists |
| Very few seams | Lower threshold to 0.3–0.4 |
| Inference slow | Normal for CPU; first run also compiles torch ops |
