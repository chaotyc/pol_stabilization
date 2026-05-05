import argparse
import json
import subprocess
import uuid
import optuna


def objective(trial, wavelength_range, epochs, loss):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dim = trial.suggest_categorical("dim", [8, 16, 32, 64])
    window_size = trial.suggest_categorical("window_size", [2, 4, 8, 16, 32, 64])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    lambda_reg_args = []
    if loss in ("RegMSE", "Angular"):
        lambda_reg = trial.suggest_float("lambda_reg", 0.01, 1.0, log=True)
        lambda_reg_args = ["--lambda-reg", str(lambda_reg)]

    run_id = f"trial_{trial.number}_{uuid.uuid4().hex[:8]}"

    cmd = [
        "python", "src/training/mamba_training.py",
        "--lr", str(lr),
        "--dim", str(dim),
        "--window-size", str(window_size),
        "--batch-size", str(batch_size),
        "--loss", loss,
        "--weight-decay", str(weight_decay),
        f"--wavelength-range={wavelength_range}",
        "--epochs", str(epochs),
        "--run-id", run_id,
        *lambda_reg_args,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Trial {trial.number} failed:\n{result.stderr[-500:]}")
        raise optuna.TrialPruned()

    metrics_path = f"results/MAMBA_test_results_{wavelength_range}_{run_id}.json"
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"Trial {trial.number}: metrics file not found at {metrics_path}")
        raise optuna.TrialPruned()

    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            trial.set_user_attr(key, val)

    return metrics["best_val_loss"]


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO sweep for Mamba training")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per trial")
    parser.add_argument("--loss", type=str, default="MSE",
                        choices=["MSE", "RegMSE", "Angular"],
                        help="Loss function to use (fixed, not tuned)")
    parser.add_argument("--wavelength-range", type=str, default="5mm")
    parser.add_argument("--study-name", type=str, default="mamba_polarization")
    parser.add_argument("--db", type=str, default="sqlite:///optuna_study.db",
                        help="Optuna storage URL (use sqlite:///path.db to persist)")
    args = parser.parse_args()

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.db,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args.wavelength_range, args.epochs, args.loss),
        n_trials=args.n_trials,
    )

    print(f"Best trial #{study.best_trial.number}")
    print(f"  Test MSE: {study.best_trial.value:.6f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print(f"\nAll results stored in: {args.db}")
    print("Visualize with: optuna-dashboard " + args.db)


if __name__ == "__main__":
    main()
