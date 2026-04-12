import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.dual_graphsage.model import DualGraphSAGE
from models.utils.dataset import compute_pos_weight, load_dataset, split_dataset
from models.utils.experiment_log import ExperimentLogger
from models.utils.losses import focal_bce_with_logits, seam_loss_with_connectivity
from models.utils.metrics import edge_f1, threshold_sweep


def _run_epoch(
    model: DualGraphSAGE,
    graphs: list[Data],
    device: torch.device,
    pos_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    lambda_conn: float = 0.0,
    focal_gamma: float = 2.0,
) -> tuple[float, dict]:
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

            if lambda_conn > 0.0:
                loss = seam_loss_with_connectivity(logits, y, edge_index, pos_weight, lambda_conn, focal_gamma)
            else:
                loss = focal_bce_with_logits(logits, y, pos_weight, focal_gamma)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.cpu())

            del x, edge_index, y, logits, loss
            torch.cuda.empty_cache()

    mean_loss = total_loss / len(graphs)
    metrics = edge_f1(torch.cat(all_logits), torch.cat(all_labels))
    return mean_loss, metrics


def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    dataset = load_dataset(args.dataset)

    train, val, test, split_info = split_dataset(dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    print(f"split — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"  train meshes: {split_info['train']}")
    print(f"  val meshes:   {split_info['val']}")
    print(f"  test meshes:  {split_info['test']}")

    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32).to(device)
        print(f"pos_weight: {pos_weight.item():.4f} (manual override)")
    else:
        pos_weight = compute_pos_weight(train).to(device)
        print(f"pos_weight: {pos_weight.item():.4f} (auto-computed)")

    model = DualGraphSAGE(
        in_dim=args.in_dim,
        hidden_dim=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    logger = ExperimentLogger(
        run_dir=args.run_dir,
        config={
            'model': 'DualGraphSAGE',
            'in_dim': args.in_dim,
            'hidden_dim': args.hidden,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'lr': args.lr,
            'lambda_conn': args.lambda_conn,
            'focal_gamma': args.focal_gamma,
            'patience': args.patience,
            'dataset': args.dataset,
            'train_graphs': len(train),
            'val_graphs': len(val),
            'test_graphs': len(test),
            'pos_weight': pos_weight.item(),
            'split': split_info,
        },
    )
    logger.log_class_balance(train, val, test)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_ctr = 0
    save_path = Path(args.run_dir) / 'best_model.pth'

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = _run_epoch(
            model, train, device, pos_weight, optimizer, args.lambda_conn, args.focal_gamma
        )
        val_loss, val_m = _run_epoch(model, val, device, pos_weight, focal_gamma=args.focal_gamma)
        epoch_time = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_m['f1'])

        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=current_lr,
            epoch_time_s=round(epoch_time, 2),
            train_f1=train_m['f1'],
            train_precision=train_m['precision'],
            train_recall=train_m['recall'],
            val_f1=val_m['f1'],
            val_precision=val_m['precision'],
            val_recall=val_m['recall'],
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}  f1 {train_m['f1']:.4f} | "
            f"val loss {val_loss:.4f}  f1 {val_m['f1']:.4f}  "
            f"prec {val_m['precision']:.4f}  rec {val_m['recall']:.4f}  "
            f"[{epoch_time:.1f}s]"
        )

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best model (val F1 = {best_val_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    print(f"\nloading best weights from {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_m = _run_epoch(model, test, device, pos_weight, focal_gamma=args.focal_gamma)
    print(
        f"test | loss {test_loss:.4f}  f1 {test_m['f1']:.4f}  "
        f"prec {test_m['precision']:.4f}  rec {test_m['recall']:.4f}  "
        f"acc {test_m['accuracy']:.4f}"
    )

    # Threshold sweep on val (select) and test (report)
    model.eval()
    val_logits, val_labels = [], []
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for data in val:
            val_logits.append(model(data.x.to(device), data.edge_index.to(device)).cpu())
            val_labels.append(data.y.cpu())
        for data in test:
            test_logits_list.append(model(data.x.to(device), data.edge_index.to(device)).cpu())
            test_labels_list.append(data.y.cpu())
    val_logits_cat = torch.cat(val_logits)
    val_labels_cat = torch.cat(val_labels)
    test_logits_cat = torch.cat(test_logits_list)
    test_labels_cat = torch.cat(test_labels_list)

    val_sweep = threshold_sweep(val_logits_cat, val_labels_cat)
    test_sweep = threshold_sweep(test_logits_cat, test_labels_cat)
    best_t = val_sweep['best']['threshold']

    print(f"\n{'─'*65}")
    print("threshold sweep (val):")
    print(f"  {'t':>5s}  {'P':>7s}  {'R':>7s}  {'F1':>7s}")
    for r in val_sweep['all']:
        marker = ' <-- best' if r['threshold'] == best_t else ''
        print(f"  {r['threshold']:>5.2f}  {r['precision']:>7.4f}  {r['recall']:>7.4f}  {r['f1']:>7.4f}{marker}")
    print("\nthreshold sweep (test):")
    print(f"  {'t':>5s}  {'P':>7s}  {'R':>7s}  {'F1':>7s}")
    for r in test_sweep['all']:
        marker = ' <-- best val' if r['threshold'] == best_t else ''
        print(f"  {r['threshold']:>5.2f}  {r['precision']:>7.4f}  {r['recall']:>7.4f}  {r['f1']:>7.4f}{marker}")
    print(f"\noptimal threshold (by val F1): {best_t:.2f}")
    print(f"{'─'*65}")

    logger.finalize(test_metrics=test_m, best_epoch=best_epoch)
    logger.save()
    logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DualGraphSAGE on dual graph for UV-seam prediction.")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument('--dataset', default='dataset_dual.pt', help='path to dual dataset')
    parser.add_argument('--run-dir', default=f'runs/dual_graphsage_{timestamp}', help='experiment output dir')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lambda-conn', type=float, default=0.0,
                        help='connectivity penalty weight (0 = disabled, try 0.1)')
    parser.add_argument('--patience', type=int, default=15, help='early-stop patience')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--in-dim', type=int, default=16, help='dual node feature dim (default: 16)')
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='override pos_weight (default: auto-computed from dataset)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='focal loss gamma (0=plain BCE, 2=standard focal)')

    main(parser.parse_args())