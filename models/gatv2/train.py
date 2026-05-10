import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.run_training import run_single_model


def main(argv=None) -> None:
    run_single_model(argv, 'gatv2')


if __name__ == '__main__':
    main(sys.argv[1:])
