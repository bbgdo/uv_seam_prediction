"""Sweep sigmoid threshold on a trained model to find optimal F1.

Usage:
    python models/utils/sweep_threshold.py \
        --dataset dataset_dual.pt \
        --weights runs/dual_graphsage_.../best_model.pth \
        --model-type graphsage
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.baselines.registry import get_baseline
from models.common.config import baseline_config
from models.utils.dataset import load_dataset, split_dataset
from models.utils.metrics import RECALL_TPR_LABEL, edge_f1


def _default_model_kwargs(model_type: str) -> dict:
    definition = get_baseline(model_type)
    config = baseline_config(model_type, definition.default_config_overrides)
    kwargs = {
        'in_dim': config.in_dim,
        'hidden_dim': config.hidden_size,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
    }
    if model_type == 'graphsage':
        kwargs.update({'aggr': config.aggr, 'skip_connections': config.skip_connections})
    elif model_type == 'gatv2':
        kwargs['heads'] = config.heads
    return kwargs


def main():
    parser = argparse.ArgumentParser(description='Sweep threshold on trained model.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--model-type', default='graphsage', choices=['graphsage', 'gatv2'])
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset(args.dataset)
    _, val, test, split_info = split_dataset(dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    print(f"val meshes:  {split_info['val']}")
    print(f"test meshes: {split_info['test']}")

    definition = get_baseline(args.model_type)
    model = definition.model_class(**_default_model_kwargs(args.model_type)).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    for split_name, graphs in [('val', val), ('test', test)]:
        all_logits, all_labels = [], []
        with torch.no_grad():
            for data in graphs:
                logits = model(data.x.to(device), data.edge_index.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(data.y.cpu())

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        probs = torch.sigmoid(logits_cat)
        pos_mask = labels_cat == 1
        neg_mask = labels_cat == 0

        print(f"\n{'='*65}")
        print(f"{split_name} set - probability distribution:")
        print(f"  seam edges    ({pos_mask.sum():>7d}):  "
              f"mean={probs[pos_mask].mean():.4f}  "
              f"median={probs[pos_mask].median():.4f}  "
              f"p10={probs[pos_mask].quantile(0.10):.4f}  "
              f"p90={probs[pos_mask].quantile(0.90):.4f}")
        print(f"  non-seam edges({neg_mask.sum():>7d}):  "
              f"mean={probs[neg_mask].mean():.4f}  "
              f"median={probs[neg_mask].median():.4f}  "
              f"p10={probs[neg_mask].quantile(0.10):.4f}  "
              f"p90={probs[neg_mask].quantile(0.90):.4f}")

        print(f"\n{split_name} threshold sweep:")
        print(
            f"  {'t':>5s}  {'P':>7s}  {RECALL_TPR_LABEL:>8s}  "
            f"{'F1':>7s}  {'TP':>7s}  {'FP':>7s}  {'FN':>7s}"
        )
        best_f1 = 0.0
        best_t = 0.5
        for t in thresholds:
            m = edge_f1(logits_cat, labels_cat, threshold=t)
            preds = (probs >= t)
            gt = labels_cat.bool()
            tp = int((preds & gt).sum())
            fp = int((preds & ~gt).sum())
            fn = int((~preds & gt).sum())
            marker = ''
            if m['f1'] > best_f1:
                best_f1 = m['f1']
                best_t = t
                marker = ' <--'
            print(f"  {t:>5.2f}  {m['precision']:>7.4f}  {m['recall']:>8.4f}  {m['f1']:>7.4f}  "
                  f"{tp:>7d}  {fp:>7d}  {fn:>7d}{marker}")

        print(f"\n  best threshold: {best_t:.2f}  F1={best_f1:.4f}")


if __name__ == '__main__':
    main()
