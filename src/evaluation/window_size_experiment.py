import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.evaluation.runner import run_training
from src.utils.runs import new_run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare SO3 input window sizes")
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[2, 4, 16, 64, 256])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--wavelength-range", default="loop_5mm")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--data-path")
    parser.add_argument("--backend", default="auto", choices=["auto", "pytorch", "cuda"])
    args = parser.parse_args(argv)
    if any(w < 2 for w in args.window_sizes):
        parser.error("All SO3 window sizes must be at least 2")
    results = [run_training(args.wavelength_range, new_run_id(f"window_{w}"), args.output_dir,
                            window_size=w, dim=args.dim, epochs=args.epochs,
                            data_path=args.data_path, backend=args.backend) for w in args.window_sizes]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(args.window_sizes, [r["test_mse"] for r in results], "o-", color="tab:red", label="Test MSE")
    ax.set(xlabel="Window size", ylabel="Test MSE", xscale="log", title="SO3 memory horizon")
    ax2 = ax.twinx()
    ax2.plot(args.window_sizes, [r["mean_fidelity"] for r in results], "s-", color="tab:blue", label="Fidelity")
    ax2.set_ylabel("Mean fidelity")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(Path(args.output_dir) / f"{new_run_id('window_size_impact')}.png")
    plt.close(fig)
    return results


if __name__ == "__main__":
    main()
