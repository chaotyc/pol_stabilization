import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.args import parse_args
from src.training.data import SParameterDataset, batch_inputs
from src.model.mamba import SO3_MODEL_TYPES
from src.model.loss import AngularLoss, PoincareRegularizedMSE, Infidelity
from src.utils.data import DATASET_PATHS, REPO_ROOT, load_stokes
from src.utils.metrics import prediction_metrics
from src.utils.plotting import output_results
from src.utils.runs import create_run_directory


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or getattr(param, "_no_weight_decay", False) or "A_log" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)


def epoch_loss(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total, samples = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in tqdm(loader, desc="Training" if optimizer else "Validation", leave=False):
            x, y = batch_inputs(x, device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            loss = criterion(model(*x), y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            samples += len(y)
    return total / samples


def train(args):
    set_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu or auto")
    if args.data_path:
        path = args.data_path
    elif args.wavelength_range in DATASET_PATHS:
        path = REPO_ROOT / DATASET_PATHS[args.wavelength_range]
    else:
        raise ValueError(f"Unknown dataset: {args.wavelength_range}")
    features, targets = load_stokes(path)
    features, targets = features[:args.max_samples], targets[:args.max_samples]
    train_end, val_end = int(0.7 * len(features)), int(0.8 * len(features))
    splits = [slice(0, train_end), slice(train_end, val_end), slice(val_end, None)]
    model_class = SO3_MODEL_TYPES[args.model_type]
    include_time = getattr(model_class, "requires_time", False)
    time_step = args.sample_interval or 1.0
    time_config = {"time_origin": 0.0, "time_scale": (train_end - 1) * time_step} if include_time else {}
    datasets = [SParameterDataset(features[s], targets[s], args.window_size, start_index=s.start or 0,
                                  include_time=include_time, time_step=time_step) for s in splits]
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0), pin_memory=device.type == "cuda")
               for i, ds in enumerate(datasets)]
    train_loader, val_loader, test_loader = loaders
    model = model_class(3, args.dim, args.layers, device=device, backend=args.backend, **time_config).to(device)
    reg = args.lambda_reg
    criterion = {
        "mse": lambda: nn.MSELoss(),
        "regmse": lambda: PoincareRegularizedMSE(0.2 if reg is None else reg),
        "angular": lambda: AngularLoss(0.02 if reg is None else reg),
        "infidelity": Infidelity,
    }[args.loss.lower()]()
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr,
    )
    run_dir = create_run_directory(args.output_dir, args.wavelength_range, args.run_id)
    checkpoint_path = run_dir / "best_model.pt"
    model_config = {
        "model_type": args.model_type, "input_dim": 3,
        "d_model": args.dim, "n_layers": args.layers, "window_size": args.window_size,
        "backend": model.backend, "wavelength_range": args.wavelength_range, "loss": args.loss,
        **time_config,
    }
    if include_time:
        model_config["time_units"] = "seconds" if args.sample_interval else "sample_index"
    training_config = {**vars(args), "run_id": run_dir.name, "data_path": str(path),
                       "num_samples": len(features), "train_end": train_end, "val_end": val_end}
    print(f"Device: {device}; backend: {model.backend}; artifacts: {run_dir}")
    best_val_loss, static_epochs = float("inf"), 0
    train_losses, val_losses = [], []
    for epoch in range(args.epochs):
        train_loss = epoch_loss(model, train_loader, criterion, device, optimizer)
        val_loss = epoch_loss(model, val_loader, criterion, device)
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError("Non-finite training or validation loss")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}: train={train_loss:.6f}, val={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss, static_epochs = val_loss, 0
            torch.save({"format_version": 1, "state_dict": model.state_dict(), "config": model_config,
                        "training_config": training_config, "best_val_loss": best_val_loss,
                        "epoch": epoch + 1}, checkpoint_path)
        else:
            static_epochs += 1
        scheduler.step(val_loss)
        if static_epochs >= 10:
            break
    metrics = {
        "wavelength_range": args.wavelength_range, "run_id": run_dir.name,
        "best_val_loss": best_val_loss, "checkpoint_path": str(checkpoint_path.resolve()),
        "config": model_config, "training_config": training_config,
        "model_info": f"{args.dim}x{args.layers}_{args.loss}_{run_dir.name}",
    }
    if not args.validation_only:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        preds, actuals = [], []
        with torch.inference_mode():
            for x, y in test_loader:
                preds.append(model(*batch_inputs(x, device)).cpu().numpy())
                actuals.append(y.numpy())
        preds, actuals = np.concatenate(preds), np.concatenate(actuals)
        metrics.update(prediction_metrics(preds, actuals))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label="Train loss")
        ax.plot(val_losses, label="Validation loss")
        ax.set(xlabel="Epoch", ylabel="Loss", title="Mamba SO3 convergence")
        ax.legend()
        fig.savefig(run_dir / "convergence.png")
        plt.close(fig)
        output_results(preds, actuals, val_end, args.window_size, metrics["model_info"],
                       n_plot=1000, output_dir=run_dir, sample_interval=args.sample_interval)
    with (run_dir / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
    print(f"Metrics saved to {run_dir / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
