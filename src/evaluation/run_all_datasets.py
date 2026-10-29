import argparse

from src.evaluation.runner import run_training
from src.utils.data import DATASET_PATHS
from src.utils.runs import new_run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train isolated SO3 runs for each dataset")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_PATHS, default=["loop_1mm", "loop_5mm", "loop_10mm", "loop_14mm"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--backend", choices=["auto", "pytorch", "cuda"], default="auto")
    args = parser.parse_args(argv)
    return [run_training(ds, new_run_id("batch_sweep"), args.output_dir, epochs=args.epochs, backend=args.backend)
            for ds in args.datasets]


if __name__ == "__main__":
    main()
