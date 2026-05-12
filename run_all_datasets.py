import subprocess
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import optuna

DATASETS = {
    "1mm":           1.0,
    "5mm":           5.0,
    "10mm":         10.0,
    "14mm":         14.0,
}

HPO_DB = "sqlite:///optuna_study.db"


def run_hpo(wavelength_range, n_trials, epochs, loss):
    study_name = f"mamba_{wavelength_range}"
    cmd = [
        sys.executable, "src/training/HPO.py",
        f"--wavelength-range={wavelength_range}",
        "--n-trials", str(n_trials),
        "--epochs", str(epochs),
        "--loss", loss,
        "--study-name", study_name,
        "--db", HPO_DB,
    ]

    print(f"\n{'=' * 60}")
    print(f"  HPO: delta_lambda = {wavelength_range}  ({n_trials} trials)")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"HPO failed for {wavelength_range} (exit code {result.returncode})")
        return None

    study = optuna.load_study(study_name=study_name, storage=HPO_DB)
    best = study.best_trial
    print(f"\nBest trial #{best.number} for {wavelength_range}:")
    print(f"  val_loss = {best.value:.6f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    return best.params


def run_final_training(wavelength_range, params, epochs, loss):
    cmd = [
        sys.executable, "src/training/mamba_training.py",
        f"--wavelength-range={wavelength_range}",
        "--lr", str(params["lr"]),
        "--dim", str(params["dim"]),
        "--window-size", str(params["window_size"]),
        "--batch-size", str(params["batch_size"]),
        "--weight-decay", str(params["weight_decay"]),
        "--loss", loss,
        "--epochs", str(epochs),
    ]
    if "lambda_reg" in params:
        cmd.extend(["--lambda-reg", str(params["lambda_reg"])])

    print(f"\n{'=' * 60}")
    print(f"  Final training: delta_lambda = {wavelength_range}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)
    return result.returncode


def load_results():
    results = {}
    for wl_name, delta_nm in DATASETS.items():
        path = f"results/MAMBA_test_results_{wl_name}.json"
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
    deltas      = [results[k]["delta_nm"] for k in sorted_keys]
    rmses       = [results[k]["test_rmse"] for k in sorted_keys]
    norm_rmses  = [results[k]["norm_test_rmse"] for k in sorted_keys]
    devs        = [results[k]["mean_deviation"] for k in sorted_keys]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(deltas, rmses, marker="o", linewidth=2, markersize=8, color="#d62728")
    axes[0].set_xlabel(r"$\Delta\lambda$ (nm)", fontsize=12)
    axes[0].set_ylabel("Test RMSE", fontsize=12)
    axes[0].set_title(r"RMSE vs $\Delta\lambda$", fontsize=13)
    axes[0].set_xticks(deltas)
    axes[0].grid(True, alpha=0.4)
    for x, y in zip(deltas, rmses):
        axes[0].annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    axes[1].plot(deltas, norm_rmses, marker="D", linewidth=2, markersize=8, color="#2ca02c")
    axes[1].set_xlabel(r"$\Delta\lambda$ (nm)", fontsize=12)
    axes[1].set_ylabel("Normalized RMSE", fontsize=12)
    axes[1].set_title(r"Norm RMSE vs $\Delta\lambda$", fontsize=13)
    axes[1].set_xticks(deltas)
    axes[1].grid(True, alpha=0.4)
    for x, y in zip(deltas, norm_rmses):
        axes[1].annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    axes[2].plot(deltas, devs, marker="s", linewidth=2, markersize=8, color="#1f77b4")
    axes[2].set_xlabel(r"$\Delta\lambda$ (nm)", fontsize=12)
    axes[2].set_ylabel("Mean Deviation from Unit Norm", fontsize=12)
    axes[2].set_title(r"Mean Deviation vs $\Delta\lambda$", fontsize=13)
    axes[2].set_xticks(deltas)
    axes[2].grid(True, alpha=0.4)
    for x, y in zip(deltas, devs):
        axes[2].annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    fig.suptitle("MAMBA Test Error vs Wavelength Separation", fontsize=15, y=1.02)
    plt.tight_layout()
    out_path = "results/MAMBA_test_error_vs_delta_lambda.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run HPO + final training for all datasets")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna trials per dataset")
    parser.add_argument("--hpo-epochs", type=int, default=100,
                        help="Max epochs per HPO trial")
    parser.add_argument("--final-epochs", type=int, default=100,
                        help="Max epochs for the final training run")
    parser.add_argument("--loss", type=str, default="MSE",
                        choices=["MSE", "RegMSE", "Angular"])
    cli_args = parser.parse_args()

    best_params = {}

    for wl_name in sorted(DATASETS, key=lambda k: DATASETS[k]):
        params = run_hpo(wl_name, cli_args.n_trials, cli_args.hpo_epochs, cli_args.loss)
        if params is None:
            print(f"Skipping {wl_name} — HPO failed.")
            continue
        best_params[wl_name] = params

    print(f"\n{'=' * 60}")
    print("  HPO complete — running final training with best params")
    print(f"{'=' * 60}")

    for wl_name, params in best_params.items():
        retcode = run_final_training(wl_name, params, cli_args.final_epochs, cli_args.loss)
        if retcode != 0:
            print(f"Final training failed for {wl_name} (exit code {retcode}), skipping.")

    results = load_results()
    if not results:
        print("No results to plot.")
        return

    print(f"\n{'=' * 60}")
    print("  Summary (best HPO params → final training)")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<10} {'RMSE':>10} {'Norm RMSE':>10} {'Mean Dev':>10} {'Fidelity':>10} {'Best Val':>10}")
    print("-" * 62)
    for k in sorted(results, key=lambda k: results[k]["delta_nm"]):
        r = results[k]
        print(f"{k:<10} {r['test_rmse']:>10.6f} {r['norm_test_rmse']:>10.6f} "
              f"{r['mean_deviation']:>10.6f} {r['mean_fidelity']:>10.6f} "
              f"{r['best_val_loss']:>10.6f}")

    plot_results(results)


if __name__ == "__main__":
    main()
