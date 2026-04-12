"""Grid search over loss hyperparameters for DualGraphSAGE.

Searches pos_weight × focal_gamma × threshold jointly.
Each combo trains for up to --epochs with early stopping on val F1
at the best threshold (not fixed 0.5).

Usage:
    python models/dual_graphsage/train_grid.py --dataset dataset_dual.pt --epochs 80
"""
import argparse
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.dual_graphsage.model import DualGraphSAGE
from models.utils.dataset import load_dataset, split_dataset
from models.utils.losses import focal_bce_with_logits
from models.utils.metrics import edge_f1


GRID = {
    'pos_weight': [5, 10, 15, 20, 30],
    'focal_gamma': [0.0, 1.0, 2.0],
}

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _best_f1_at_any_threshold(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Return (best_f1, best_threshold) across THRESHOLDS."""
    best_f1, best_t = 0.0, 0.5
    for t in THRESHOLDS:
        m = edge_f1(logits, labels, threshold=t)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_t = t
    return best_f1, best_t


def _run_epoch(
    model: DualGraphSAGE,
    graphs: list[Data],
    device: torch.device,
    pos_weight_tensor: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    focal_gamma: float = 2.0,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_logits, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for data in graphs:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)
            logits = model(x, edge_index)
            loss = focal_bce_with_logits(logits, y, pos_weight_tensor, focal_gamma)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.cpu())

            del x, edge_index, y, logits, loss
            torch.cuda.empty_cache()

    return total_loss / len(graphs), torch.cat(all_logits), torch.cat(all_labels)


def run_one_config(
    train: list[Data],
    val: list[Data],
    test: list[Data],
    device: torch.device,
    pw: float,
    gamma: float,
    epochs: int,
    patience: int,
    lr: float,
) -> dict:
    pw_tensor = torch.tensor([pw], dtype=torch.float32).to(device)
    model = DualGraphSAGE(in_dim=16, hidden_dim=128, num_layers=3, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_f1 = 0.0
    best_threshold = 0.5
    best_state = None
    patience_ctr = 0
    epoch = 0

    for epoch in range(1, epochs + 1):
        _run_epoch(model, train, device, pw_tensor, optimizer, gamma)
        _, val_logits, val_labels = _run_epoch(model, val, device, pw_tensor, focal_gamma=gamma)

        val_f1, val_t = _best_f1_at_any_threshold(val_logits, val_labels)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = val_t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, test_logits, test_labels = _run_epoch(model, test, device, pw_tensor, focal_gamma=gamma)
    test_m = edge_f1(test_logits, test_labels, threshold=best_threshold)
    test_best_f1, test_best_t = _best_f1_at_any_threshold(test_logits, test_labels)

    return {
        'pos_weight': pw,
        'focal_gamma': gamma,
        'best_val_f1': round(best_val_f1, 4),
        'best_threshold': best_threshold,
        'test_f1_at_val_threshold': round(test_m['f1'], 4),
        'test_precision': round(test_m['precision'], 4),
        'test_recall': round(test_m['recall'], 4),
        'test_best_f1': round(test_best_f1, 4),
        'test_best_threshold': test_best_t,
        'epochs_trained': epoch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Grid search over loss hyperparameters.')
    parser.add_argument('--dataset', default='dataset_dual.pt')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    dataset = load_dataset(args.dataset)
    train, val, test, split_info = split_dataset(dataset)
    print(f'split: train={len(train)}, val={len(val)}, test={len(test)}')
    print(f'val meshes:  {split_info["val"]}')
    print(f'test meshes: {split_info["test"]}')

    configs = list(product(GRID['pos_weight'], GRID['focal_gamma']))
    results = []

    print(f'\nrunning {len(configs)} configurations ({args.epochs} epochs each, patience={args.patience})...\n')
    print(f'{"pw":>5s}  {"gamma":>5s}  {"val_F1":>7s}  {"thr":>5s}  '
          f'{"test_F1":>7s}  {"P":>7s}  {"R":>7s}  {"ep":>4s}  {"time":>6s}')
    print('-' * 70)

    for pw, gamma in configs:
        t0 = time.time()
        result = run_one_config(train, val, test, device, pw, gamma, args.epochs, args.patience, args.lr)
        elapsed = time.time() - t0
        results.append(result)

        print(f'{pw:>5.0f}  {gamma:>5.1f}  {result["best_val_f1"]:>7.4f}  '
              f'{result["best_threshold"]:>5.2f}  {result["test_f1_at_val_threshold"]:>7.4f}  '
              f'{result["test_precision"]:>7.4f}  {result["test_recall"]:>7.4f}  '
              f'{result["epochs_trained"]:>4d}  {elapsed:>5.0f}s')

    results.sort(key=lambda r: r['best_val_f1'], reverse=True)

    print(f'\n{"="*70}')
    print('TOP 5 by val F1:')
    for i, r in enumerate(results[:5]):
        print(f'  {i+1}. pw={r["pos_weight"]}, gamma={r["focal_gamma"]}, '
              f'val_F1={r["best_val_f1"]:.4f} (t={r["best_threshold"]:.2f}), '
              f'test_F1={r["test_f1_at_val_threshold"]:.4f} '
              f'(P={r["test_precision"]:.4f} R={r["test_recall"]:.4f})')

    best = results[0]
    print(f'\nrecommended next run:')
    print(f'  python models/dual_graphsage/train.py --dataset {args.dataset} '
          f'--epochs 200 --patience 20 '
          f'--pos-weight {best["pos_weight"]} --focal-gamma {best["focal_gamma"]}')

    out_dir = Path(f'runs/grid_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'grid_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nresults saved to {out_path}')


if __name__ == '__main__':
    main()
