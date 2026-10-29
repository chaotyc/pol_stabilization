import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import optuna

from src.evaluation.runner import run_training, study_name
from src.utils.data import WAVELENGTHS
from src.utils.runs import new_run_id

DATASETS = WAVELENGTHS
HPO_DB = "sqlite:///optuna_study.db"


def load_best_params(wavelength_range, db, loss="infidelity", protocol=None):
    name = study_name(wavelength_range, loss)
    try:
        study = optuna.load_study(study_name=name, storage=db)
    except KeyError:
        print(f"No study found for {name}; run python -m src.evaluation.hpo first")
        return None
    if protocol is not None and study.user_attrs.get("protocol") != protocol:
        raise ValueError(f"{name} uses a different training protocol; match the HPO data, seed, and sample limit")
    try:
        return study.best_trial.params
    except ValueError:
        print(f"No completed trials for {name}")
        return None


def plot_results(results, output_dir="results"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["delta_nm"])
    deltas      = [results[k]["delta_nm"] for k in sorted_keys]
    rmses       = [results[k]["test_rmse"] for k in sorted_keys]
    norm_rmses  = [results[k]["norm_test_rmse"] for k in sorted_keys]
    devs        = [results[k]["mean_deviation"] for k in sorted_keys]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for regime in ("synthetic", "loop"):
        indices = [i for i, k in enumerate(sorted_keys) if k.startswith(regime)]
        if indices:
            axes[0].plot([deltas[i] for i in indices], [rmses[i] for i in indices], marker="o", label=regime)
    axes[0].legend()
    axes[0].set_xlabel(r"$\Delta\lambda$ (nm)", fontsize=12)
    axes[0].set_ylabel("Test RMSE", fontsize=12)
    axes[0].set_title(r"RMSE vs $\Delta\lambda$", fontsize=13)
    axes[0].set_xticks(deltas)
    axes[0].grid(True, alpha=0.4)
    for x, y in zip(deltas, rmses):
        axes[0].annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    for regime in ("synthetic", "loop"):
        indices = [i for i, k in enumerate(sorted_keys) if k.startswith(regime)]
        if indices:
            axes[1].plot([deltas[i] for i in indices], [norm_rmses[i] for i in indices], marker="D", label=regime)
    axes[1].legend()
    axes[1].set_xlabel(r"$\Delta\lambda$ (nm)", fontsize=12)
    axes[1].set_ylabel("Normalized RMSE", fontsize=12)
    axes[1].set_title(r"Norm RMSE vs $\Delta\lambda$", fontsize=13)
    axes[1].set_xticks(deltas)
    axes[1].grid(True, alpha=0.4)
    for x, y in zip(deltas, norm_rmses):
        axes[1].annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    for regime in ("synthetic", "loop"):
        indices = [i for i, k in enumerate(sorted_keys) if k.startswith(regime)]
        if indices:
            axes[2].plot([deltas[i] for i in indices], [devs[i] for i in indices], marker="s", label=regime)
    axes[2].legend()
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
    out_path = Path(output_dir) / f"{new_run_id('test_error_vs_delta_lambda')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrain the best validation-selected SO3 configurations")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--loss", type=str.lower, default="infidelity", choices=["mse", "regmse", "angular", "infidelity"])
    parser.add_argument("--db", default=HPO_DB)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=300000)
    parser.add_argument("--data-path")
    parser.add_argument("--backend", choices=["auto", "pytorch", "cuda"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    protocol = {"model_type": "PolarizationMambaSO3", "loss": args.loss, "seed": args.seed,
                "max_samples": args.max_samples, "data_path": str(Path(args.data_path).resolve()) if args.data_path else None}
    results = {}
    for ds in args.datasets:
        params = load_best_params(ds, args.db, args.loss, protocol)
        if params is None:
            continue
        data = run_training(ds, new_run_id(f"final_{args.loss}"), args.output_dir, **params,
                            epochs=args.epochs, loss=args.loss, seed=args.seed, max_samples=args.max_samples,
                            data_path=args.data_path, backend=args.backend, device=args.device)
        data["delta_nm"] = DATASETS[ds]
        results[ds] = data
        print(f"{ds}: RMSE={data['test_rmse']:.6f}, normalized RMSE={data['norm_test_rmse']:.6f}, fidelity={data['mean_fidelity']:.6f}")
    if results:
        plot_results(results, args.output_dir)
    else:
        raise RuntimeError("No completed HPO studies found; run python -m src.evaluation.hpo first")
    return results


if __name__ == "__main__":
    main()
