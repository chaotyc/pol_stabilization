import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.evaluation.runner import run_training
from src.utils.data import WAVELENGTHS
from src.utils.runs import new_run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare wavelength separations using the SO3 model")
    parser.add_argument("--datasets", nargs="+", choices=WAVELENGTHS, default=["loop_1mm", "loop_5mm", "loop_10mm", "loop_14mm"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--backend", default="auto", choices=["auto", "pytorch", "cuda"])
    args = parser.parse_args(argv)
    results = [run_training(ds, new_run_id("wavelength"), args.output_dir, window_size=args.window_size,
                            dim=args.dim, epochs=args.epochs, backend=args.backend) for ds in args.datasets]
    fig, ax = plt.subplots(figsize=(10, 6))
    for regime in ("synthetic", "loop"):
        selected = sorted([r for r in results if r["wavelength_range"].startswith(regime)],
                          key=lambda r: WAVELENGTHS[r["wavelength_range"]])
        if selected:
            ax.plot([WAVELENGTHS[r["wavelength_range"]] for r in selected], [r["test_rmse"] for r in selected], "o-", label=regime)
    ax.set(xlabel="Wavelength separation (nm)", ylabel="Test RMSE", title="SO3 error vs wavelength separation")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(Path(args.output_dir) / f"{new_run_id('delta_lambda_impact')}.png")
    plt.close(fig)
    return results


if __name__ == "__main__":
    main()
