import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.baseline_train import (  # noqa: F401
    dataset_metadata_summary,
    set_random_seeds,
    train_baseline,
)
from tools.run_baseline import _fill_model_defaults, main as run_baseline_main


def main(args: argparse.Namespace | list[str] | None = None) -> None:
    if isinstance(args, argparse.Namespace):
        args.model = getattr(args, 'model', None) or 'graphsage'
        train_baseline(_fill_model_defaults(args))
        return
    run_baseline_main(args, default_model='graphsage')


if __name__ == '__main__':
    print('note: tools/run_baseline.py --model graphsage is the canonical baseline runner.')
    main(sys.argv[1:])
