import os
import sys
import argparse

_RUN_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _RUN_LOCAL_DIR not in sys.path:
    sys.path.insert(0, _RUN_LOCAL_DIR)

import common
from experiments import main_result, oversmoothing, ablation


def build_parser():
    parser = argparse.ArgumentParser(description="Local GNN over-smoothing experiments.")
    parser.add_argument(
        "--experiment", "-e",
        choices=["all", "main", "oversmoothing", "ablation"],
        default="all",
    )
    parser.add_argument(
        "--results-dir", "-o",
        default=common.RESULTS_DIR,
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    results_dir = common.ensure_dir(args.results_dir)
    print(f"Device: {common.DEVICE}")
    print(f"Results dir: {results_dir}\n")

    if args.experiment in ("all", "main"):
        main_result.run(results_dir=results_dir)

    if args.experiment in ("all", "oversmoothing"):
        oversmoothing.run(results_dir=results_dir)

    if args.experiment in ("all", "ablation"):
        ablation.run(results_dir=results_dir)

    print("\nDone. Artifacts written under:", results_dir)


if __name__ == "__main__":
    main()
