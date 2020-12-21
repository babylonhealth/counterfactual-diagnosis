from argparse import ArgumentParser
from pathlib import Path
from sys import exit

from constants import DATA_PATH
from experiments import run_vignettes_experiment
from inference import create_marginals_files
from results import produce_results
from utils import ensure_dir, load_from_json


def parse_args():
    parser = ArgumentParser(description="Counterfactual Diagnosis experimental code")
    parser.add_argument(
        "--datapath", type=Path, default=DATA_PATH, help="Path to data files"
    )
    parser.add_argument(
        "--first", type=int, required=False, default=None, help="Run the first N cards"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Output path for results files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Run with verbose logging",
    )
    parser.add_argument(
        "--reproduce",
        default=False,
        action="store_true",
        help="Reproduce paper results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ensure_dir(args.results)

    if args.reproduce is False:
        print(
            f">>> Running over local model files in {args.datapath/'test_networks.json'}"
        )
        create_marginals_files(args=args)

    run_vignettes_experiment(args=args)

    if args.reproduce is True:
        produce_results(args=args)
