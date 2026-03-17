import subprocess
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

DATASETS = {
    "-5mm":  -5.0,
    "1mm":    1.0,
    "5mm":    5.0,
    "10mm":  10.0,
    "14mm":  14.0,
}


def run_training(wavelength_range, extra_args=None):
    cmd = [sys.executable, "mamba_training.py", f"--wavelength-range={wavelength_range}"]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'=' * 60}")
    print(f"  Training: delta_lambda = {wavelength_range}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)
    return result.returncode


def load_results():
    results = {}
    for wl_name, delta_nm in DATASETS.items():
        path = f"Results/MAMBA_test_results_{wl_name}.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            data["delta_nm"] = delta_nm
            results[wl_name] = data
        else:
            print(f"Warning: no results found for {wl_name} ({path})")
    return results


def plot_results(results):
    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["delta_nm"])
    deltas = [results[k]["delta_nm"] for k in sorted_keys]
    rmses  = [results[k]["test_rmse"] for k in sorted_keys]
    devs   = [results[k]["mean_deviation"] for k in sorted_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(deltas, rmses, marker="o", linewidth=2, markersize=8, color="#d62728")
    ax1.set_xlabel(r"$\Delta\lambda$ (nm)", fontsize=12)
    ax1.set_ylabel("Test RMSE", fontsize=12)
    ax1.set_title(r"Mean RMSE vs $\Delta\lambda$", fontsize=13)
    ax1.set_xticks(deltas)
    ax1.grid(True, alpha=0.4)
    for x, y in zip(deltas, rmses):
        ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    ax2.plot(deltas, devs, marker="s", linewidth=2, markersize=8, color="#1f77b4")
    ax2.set_xlabel(r"$\Delta\lambda$ (nm)", fontsize=12)
    ax2.set_ylabel("Mean Deviation from Unit Norm", fontsize=12)
    ax2.set_title(r"Mean Deviation vs $\Delta\lambda$", fontsize=13)
    ax2.set_xticks(deltas)
    ax2.grid(True, alpha=0.4)
    for x, y in zip(deltas, devs):
        ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    fig.suptitle("MAMBA Test Error vs Wavelength Separation", fontsize=15, y=1.02)
    plt.tight_layout()
    out_path = "Results/MAMBA_test_error_vs_delta_lambda.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


def main():
    extra_args = sys.argv[1:]

    for wl_name in sorted(DATASETS, key=lambda k: DATASETS[k]):
        retcode = run_training(wl_name, extra_args)
        if retcode != 0:
            print(f"Training failed for {wl_name} (exit code {retcode}), skipping.")

    results = load_results()
    if not results:
        print("No results to plot.")
        return

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<10} {'RMSE':>10} {'Mean Dev':>10} {'Best Val':>10}")
    print("-" * 42)
    for k in sorted(results, key=lambda k: results[k]["delta_nm"]):
        r = results[k]
        print(f"{k:<10} {r['test_rmse']:>10.6f} {r['mean_deviation']:>10.6f} "
              f"{r['best_val_loss']:>10.6f}")

    plot_results(results)


if __name__ == "__main__":
    main()
