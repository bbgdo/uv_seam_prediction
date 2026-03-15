import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

# allow running as a script from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.graphsage.model import UVSeamGNN
from models.utils.dataset import compute_pos_weight, load_dataset, split_dataset
from models.utils.metrics import edge_f1


def _run_epoch(
    model: UVSeamGNN,
    graphs: list[Data],
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict]:
    """Single pass over a list of graphs. Returns (mean_loss, mean_metrics)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_logits, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for data in graphs:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            y = data.y.to(device)

            logits = model(x, edge_index, edge_attr)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.cpu())

            # explicitly free GPU tensors — Python's GC isn't eager enough
            # and activations accumulate across graph iterations
            del x, edge_index, edge_attr, y, logits, loss
            torch.cuda.empty_cache()

    mean_loss = total_loss / len(graphs)
    metrics = edge_f1(torch.cat(all_logits), torch.cat(all_labels))
    return mean_loss, metrics


def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    dataset = load_dataset(args.dataset)

    if args.max_edges:
        before = len(dataset)
        dataset = [d for d in dataset if d.edge_index.shape[1] <= args.max_edges]
        skipped = before - len(dataset)
        if skipped:
            print(f"skipped {skipped} graph(s) exceeding --max-edges {args.max_edges}")

    train, val, test = split_dataset(dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    print(f"split — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    pos_weight = compute_pos_weight(train).to(device)
    print(f"pos_weight: {pos_weight.item():.4f}")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = UVSeamGNN(
        node_in_dim=args.node_dim,
        edge_in_dim=args.edge_dim,
        hidden_dim=args.hidden,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # halve LR if val F1 stagnates for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_f1 = 0.0
    patience_ctr = 0
    save_path = Path(args.save_dir) / 'best_model.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_m = _run_epoch(model, train, criterion, device, optimizer)
        val_loss, val_m = _run_epoch(model, val, criterion, device)

        scheduler.step(val_m['f1'])

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}  f1 {train_m['f1']:.4f} | "
            f"val loss {val_loss:.4f}  f1 {val_m['f1']:.4f}  "
            f"prec {val_m['precision']:.4f}  rec {val_m['recall']:.4f}"
        )

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ saved best model  (val F1 = {best_val_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    print(f"\nloading best weights from {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_m = _run_epoch(model, test, criterion, device)
    print(
        f"test | loss {test_loss:.4f}  f1 {test_m['f1']:.4f}  "
        f"prec {test_m['precision']:.4f}  rec {test_m['recall']:.4f}  "
        f"acc {test_m['accuracy']:.4f}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train UV-seam GNN.")

    parser.add_argument('--dataset', default='dataset.pt', help='path to dataset.pt')
    parser.add_argument('--save-dir', default='models/graphsage', help='where to save best_model.pth')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=15, help='early-stop patience')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--node-dim', type=int, default=6, help='node feature dim (default: 6)')
    parser.add_argument('--edge-dim', type=int, default=4, help='edge feature dim (default: 4)')
    parser.add_argument('--max-edges', type=int, default=2_000_000,
                        help='skip graphs with more directed edges than this (default: 2M)')

    main(parser.parse_args())
