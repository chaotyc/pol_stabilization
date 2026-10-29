"""Validation-only hyperparameter optimization for the current SO3 model."""
import argparse
import json
from pathlib import Path

import optuna

from src.evaluation.runner import run_training, study_name
from src.utils.data import DATASET_PATHS
from src.utils.runs import new_run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength-range", choices=DATASET_PATHS, default="loop_5mm")
    parser.add_argument("--loss", type=str.lower, choices=["mse", "regmse", "angular", "infidelity"], default="infidelity")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--db", default="sqlite:///optuna_study.db")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=300000)
    parser.add_argument("--data-path")
    parser.add_argument("--backend", choices=["auto", "pytorch", "cuda"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    if args.n_trials < 1 or args.epochs < 1:
        parser.error("--n-trials and --epochs must be positive")
    name = study_name(args.wavelength_range, args.loss)
    study = optuna.create_study(study_name=name, storage=args.db, direction="minimize", load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(seed=args.seed))
    protocol = {"model_type": "PolarizationMambaSO3", "loss": args.loss, "seed": args.seed,
                "max_samples": args.max_samples, "data_path": str(Path(args.data_path).resolve()) if args.data_path else None}
    previous = study.user_attrs.get("protocol")
    if previous is not None and previous != protocol:
        raise ValueError("Existing study uses a different training protocol; select a new --db")
    study.set_user_attr("protocol", protocol)

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dim": trial.suggest_categorical("dim", [16, 32, 64]),
            "layers": trial.suggest_int("layers", 2, 4),
            "window_size": trial.suggest_categorical("window_size", [2, 4, 16, 64, 128, 256]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }
        if args.loss in ("regmse", "angular"):
            params["lambda_reg"] = trial.suggest_float("lambda_reg", 1e-4, 1.0, log=True)
        run_id = new_run_id(f"hpo_{args.loss}_{trial.number}")
        metrics = run_training(args.wavelength_range, run_id, args.output_dir, **params,
                               epochs=args.epochs, loss=args.loss, seed=args.seed, max_samples=args.max_samples,
                               data_path=args.data_path, backend=args.backend, device=args.device, validation_only=True)
        trial.set_user_attr("checkpoint_path", metrics["checkpoint_path"])
        return metrics["best_val_loss"]

    study.optimize(objective, n_trials=args.n_trials)
    output = Path(args.output_dir) / "hpo"
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{name}_best.json").open("w") as stream:
        json.dump({"study_name": name, "protocol": protocol, "params": study.best_params,
                   "best_val_loss": study.best_value}, stream, indent=2)
    print(f"Best validation loss: {study.best_value:.6f}; parameters: {study.best_params}")
    return study


if __name__ == "__main__":
    main()
