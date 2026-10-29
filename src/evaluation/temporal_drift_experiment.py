"""Compare frozen, time-conditioned, and periodically adapted six-feature models.

The defaults fit on samples [0, 300000) and evaluate every target in
[300000, 2000000). Test windows may use past TxP context from the training
prefix. An adaptive model sees each target only AFTER its prediction is scored.
"""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.model.loss import Infidelity
from src.model.mamba import PolarizationMambaSO3Combined, PolarizationMambaSO3CombinedTime
from src.training.data import SParameterDataset, batch_inputs
from src.training.mamba_training import make_optimizer, set_seed
from src.utils.data import REPO_ROOT, load_stokes
from src.utils.metrics import prediction_metrics
from src.utils.runs import create_run_directory


METHODS = ("six_features", "six_features_time", "six_features_adaptive")
LABELS = {"six_features": "Six features (frozen)", "six_features_time": "Six features + time",
          "six_features_adaptive": "Six features + periodic adaptation"}


def parse_experiment_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data/chicago_loop/txp_1551.5_pax_1556.5_fiber_loop.mat"))
    parser.add_argument("--total-samples", type=int, default=2_000_000)
    parser.add_argument("--train-samples", type=int, default=300_000)
    parser.add_argument("--epochs", type=int, default=20, help="Fixed training budget; no test-based model selection")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--metric-block-size", type=int, default=10_000)
    parser.add_argument("--adapt-every", type=int, default=10_000)
    parser.add_argument("--adapt-window", type=int, default=20_000)
    parser.add_argument("--adapt-steps", type=int, default=20)
    parser.add_argument("--adapt-batch-size", type=int, default=128)
    parser.add_argument("--adapt-lr", type=float, default=1e-4)
    parser.add_argument("--replay-fraction", type=float, default=0.2)
    parser.add_argument("--sample-interval", type=float, help="Seconds per sample, if known; otherwise time is the sample index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("auto", "pytorch", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (1 avoids overhead on small models)")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    for key in ("total_samples", "train_samples", "epochs", "dim", "layers", "batch_size", "eval_batch_size",
                "metric_block_size", "adapt_every", "adapt_window", "adapt_steps", "adapt_batch_size", "threads"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not 2 <= args.window_size < args.train_samples < args.total_samples:
        parser.error("Require 2 <= window-size < train-samples < total-samples")
    for key in ("lr", "adapt_lr"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    if not math.isfinite(args.replay_fraction) or not 0 <= args.replay_fraction < 1:
        parser.error("--replay-fraction must be in [0, 1)")
    if args.sample_interval is not None and (not math.isfinite(args.sample_interval) or args.sample_interval <= 0):
        parser.error("--sample-interval must be finite and positive")
    return args


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_config(model, args):
    config = {"model_type": type(model).__name__, "input_dim": 3, "d_model": args.dim,
              "n_layers": args.layers, "window_size": args.window_size, "backend": model.backend}
    if getattr(model, "requires_time", False):
        config.update(time_origin=float(model.time_origin), time_scale=float(model.time_scale),
                      time_units="seconds" if args.sample_interval else "sample_index")
    return config


def save_checkpoint(path, model, args, **metadata):
    torch.save({"format_version": 1, "config": model_config(model, args), "state_dict": model.state_dict(),
                "training_config": vars(args), **metadata}, path)


def train_initial(features, targets, args, device, run_dir, method, training_rows):
    """Fit using ALL of the initial prefix and a predeclared, fixed epoch budget."""
    set_seed(args.seed)
    timed = method == "six_features_time"
    kwargs = {"time_origin": 0.0, "time_scale": (args.train_samples - 1) * (args.sample_interval or 1.0)} if timed else {}
    cls = PolarizationMambaSO3CombinedTime if timed else PolarizationMambaSO3Combined
    model = cls(3, args.dim, args.layers, device=device, backend=args.backend, **kwargs).to(device)
    dataset = SParameterDataset(features[:args.train_samples], targets[:args.train_samples], args.window_size,
                                include_time=timed, time_step=args.sample_interval or 1.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type == "cuda")
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    criterion = Infidelity()
    for epoch in range(args.epochs):
        model.train()
        synchronize(device)
        start = time.perf_counter()
        total_loss = 0.0
        for inputs, y in tqdm(loader, desc=f"{method} {epoch + 1}/{args.epochs}", disable=not sys.stderr.isatty()):
            inputs, y = batch_inputs(inputs, device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(*inputs), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss for {method}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        synchronize(device)
        row = {"method": method, "epoch": epoch + 1, "train_infidelity": total_loss / len(dataset),
               "seconds": time.perf_counter() - start, "training_windows": len(dataset)}
        training_rows.append(row)
        print(f"{method}: epoch {epoch + 1}/{args.epochs}, infidelity={row['train_infidelity']:.6f}, "
              f"seconds={row['seconds']:.1f}", flush=True)
        write_csv(run_dir / "training.csv", training_rows)
        save_checkpoint(run_dir / f"{method}_initial.pt", model, args,
                        trained_through_index_exclusive=args.train_samples, epoch=epoch + 1)
    return model.eval()


class PeriodicHeadAdapter:
    """Update only the rotation head, using scored recent labels plus initial replay."""

    def __init__(self, model, dataset, args, device):
        self.model, self.dataset, self.args, self.device = model, dataset, args, device
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("head."))
        self.optimizer = make_optimizer(model, args.adapt_lr, args.weight_decay)
        self.rng = np.random.default_rng(args.seed + 1)
        self.observed_end = args.train_samples

    def update(self, observed_end):
        if not self.observed_end < observed_end <= len(self.dataset.features):
            raise ValueError("Adaptation boundary must advance within the observed data")
        args = self.args
        # Online labels are available only up to observed_end - 1.
        recent_start = max(args.train_samples, observed_end - args.adapt_window)
        recent = np.arange(recent_start, observed_end)
        weights = np.exp(3 * (recent - recent[-1]) / args.adapt_window)
        weights /= weights.sum()
        replay_count = min(args.adapt_batch_size - 1, int(args.adapt_batch_size * args.replay_fraction))
        recent_count = args.adapt_batch_size - replay_count
        min_used, max_used, total_loss = observed_end, -1, 0.0
        self.model.eval()  # Frozen backbone remains in inference mode; the head still receives gradients.
        synchronize(self.device)
        start = time.perf_counter()
        for _ in range(args.adapt_steps):
            recent_ids = self.rng.choice(recent, size=recent_count, replace=True, p=weights)
            replay_ids = self.rng.integers(args.window_size, args.train_samples, size=replay_count)
            indices = np.concatenate((recent_ids, replay_ids))
            min_used, max_used = min(min_used, int(indices.min())), max(max_used, int(indices.max()))
            x = torch.stack([self.dataset.features[i - args.window_size:i] for i in indices]).to(self.device)
            y = self.dataset.targets[torch.as_tensor(indices)].unsqueeze(1).to(self.device)
            self.optimizer.zero_grad()
            loss = Infidelity()(self.model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite adaptation loss")
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        synchronize(self.device)
        self.observed_end = observed_end
        return {"observed_end_exclusive": observed_end, "applies_from_target_index": observed_end,
                "recent_start": recent_start, "recent_end_exclusive": observed_end,
                "replay_start": args.window_size, "replay_end_exclusive": args.train_samples,
                "min_target_used": min_used, "max_target_used": max_used,
                "steps": args.adapt_steps, "recent_samples_per_step": recent_count,
                "replay_samples_per_step": replay_count, "mean_update_infidelity": total_loss / args.adapt_steps,
                "seconds": time.perf_counter() - start}


def predict_range(model, dataset, start, end, args, device):
    """Predict target t from TxP[t-window_size:t], including across test-block boundaries."""
    subset = Subset(dataset, range(start - args.window_size, end - args.window_size))
    loader = DataLoader(subset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=device.type == "cuda")
    predictions = []
    model.eval()
    with torch.inference_mode():
        for inputs, _ in loader:
            predictions.append(model(*batch_inputs(inputs, device)).cpu().numpy().reshape(-1, 3))
    return np.concatenate(predictions)


def temporal_metrics(predictions, actuals):
    # Compute angle in float64 so small errors aren't rounded to zero by float32 dot products.
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(actuals, dtype=np.float64)
    p = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-12)
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(p, y), axis=-1), np.sum(p * y, axis=-1)))
    return {**prediction_metrics(predictions, actuals), "mean_angular_error_deg": float(np.mean(angles)),
            "median_angular_error_deg": float(np.median(angles)), "p95_angular_error_deg": float(np.percentile(angles, 95))}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_chronologically(models, features, targets, args, device, run_dir):
    dataset = SParameterDataset(features, targets, args.window_size)
    timed_dataset = SParameterDataset(features, targets, args.window_size, include_time=True,
                                     time_step=args.sample_interval or 1.0)
    adapter = PeriodicHeadAdapter(models["six_features_adaptive"], dataset, args, device)
    rows, updates = [], []
    for block_start in range(args.train_samples, args.total_samples, args.metric_block_size):
        block_end = min(block_start + args.metric_block_size, args.total_samples)
        predictions = {method: [] for method in METHODS}
        prediction_seconds = {method: 0.0 for method in METHODS}
        position = block_start
        while position < block_end:
            next_update = args.train_samples + ((position - args.train_samples) // args.adapt_every + 1) * args.adapt_every
            end = min(block_end, next_update)
            for method in METHODS:
                synchronize(device)
                start = time.perf_counter()
                predictions[method].append(predict_range(models[method], timed_dataset if method == "six_features_time" else dataset,
                                                         position, end, args, device))
                synchronize(device)
                prediction_seconds[method] += time.perf_counter() - start
            # Every method has already predicted this segment before the adaptive model sees its labels.
            if end == next_update and end < args.total_samples:
                update = adapter.update(end)
                updates.append(update)
                with (run_dir / "adaptation.jsonl").open("a") as stream:
                    stream.write(json.dumps(update, allow_nan=False) + "\n")
            position = end
        for method in METHODS:
            metrics = temporal_metrics(np.concatenate(predictions[method]), targets[block_start:block_end])
            row = {"method": method, "start_index": block_start, "end_index_exclusive": block_end,
                   "n_samples": block_end - block_start,
                   "samples_since_training": block_start - args.train_samples,
                   "midpoint_index": (block_start + block_end - 1) / 2,
                   "prediction_seconds": prediction_seconds[method], **metrics}
            if args.sample_interval:
                row.update(start_seconds=block_start * args.sample_interval,
                           end_seconds_exclusive=block_end * args.sample_interval)
            rows.append(row)
        write_csv(run_dir / "metrics_by_time.csv", rows)
        display = ", ".join(f"{method}={rows[-3 + i]['mean_angular_error_deg']:.3f}°" for i, method in enumerate(METHODS))
        print(f"Scored targets [{block_start:,}, {block_end:,}): {display}", flush=True)
    save_checkpoint(run_dir / "six_features_adaptive_final.pt", adapter.model, args,
                    trained_through_index_exclusive=adapter.observed_end, adaptation_updates=len(updates))
    return rows, updates


def summarize(rows, updates, args):
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        count = sum(row["n_samples"] for row in selected)
        means = {key: sum(row[key] * row["n_samples"] for row in selected) / count
                 for key in ("test_mse", "norm_test_mse", "mean_fidelity", "mean_angular_error_deg")}
        summary[method] = {"test_samples": count, **means, "test_rmse": math.sqrt(means["test_mse"]),
                           "norm_test_rmse": math.sqrt(means["norm_test_mse"]),
                           "first_block_angular_error_deg": selected[0]["mean_angular_error_deg"],
                           "last_block_angular_error_deg": selected[-1]["mean_angular_error_deg"],
                           "prediction_seconds": sum(row["prediction_seconds"] for row in selected)}
    summary["adaptation"] = {"updates": len(updates), "seconds": sum(row["seconds"] for row in updates),
                             "last_observed_end_exclusive": updates[-1]["observed_end_exclusive"] if updates else args.train_samples}
    return summary


def plot_evolution(rows, args, run_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    plots = (("mean_angular_error_deg", "Mean angular error (degrees)"),
             ("p95_angular_error_deg", "95th percentile angular error (degrees)"),
             ("mean_fidelity", "Mean fidelity"), ("norm_test_rmse", "Normalized prediction RMSE"))
    scale = args.sample_interval / 3600 if args.sample_interval else 1e-6
    for ax, (key, label) in zip(axes.flat, plots):
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method]
            ax.plot([row["midpoint_index"] * scale for row in selected], [row[key] for row in selected],
                    label=LABELS[method], linewidth=1.6)
        ax.axvspan(0, args.train_samples * scale, color="gray", alpha=0.12)
        ax.axvline(args.train_samples * scale, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel(label)
        ax.set_xlim(0, args.total_samples * scale)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Elapsed time (hours)" if args.sample_interval else "Sample index (millions)")
    fig.suptitle(f"Temporal generalization: first {args.train_samples:,} samples for initial training; "
                 f"{args.metric_block_size:,}-sample evaluation blocks", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(run_dir / "performance_over_time.png", dpi=180)
    fig.savefig(run_dir / "performance_over_time.svg")
    plt.close(fig)


def main(argv=None):
    args = parse_experiment_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    features, targets = load_stokes(args.data_path)
    source_samples = len(features)
    if source_samples < args.total_samples:
        raise ValueError(f"Requested {args.total_samples:,} samples, but dataset has only {source_samples:,}")
    features = np.ascontiguousarray(features[:args.total_samples])
    targets = np.ascontiguousarray(targets[:args.total_samples])
    run_dir = create_run_directory(args.output_dir, "temporal_drift", args.run_id)
    protocol = {**vars(args), "data_path": str(Path(args.data_path).resolve()), "source_samples": source_samples,
                "device": str(device), "torch_version": str(torch.__version__), "indexing": "zero-based, end-exclusive",
                "training_targets": [args.window_size, args.train_samples],
                "test_targets": [args.train_samples, args.total_samples], "objective": "infidelity",
                "time_units": "seconds" if args.sample_interval else "sample_index",
                "time_origin": 0.0, "time_scale": (args.train_samples - 1) * (args.sample_interval or 1.0),
                "model_selection": "fixed epochs; all initial training samples used; no test-based selection",
                "feedback": "PAX labels available after scoring; no updates before first adapt-every test samples",
                "adaptation": "head only; exponential recent-sample weighting plus uniform initial-prefix replay",
                "initialization": "same backbone/head initialization and minibatch order; time coefficient starts at zero"}
    (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    print(f"Artifacts: {run_dir.resolve()}", flush=True)
    training_rows = []
    baseline = train_initial(features, targets, args, device, run_dir, "six_features", training_rows)
    timed = train_initial(features, targets, args, device, run_dir, "six_features_time", training_rows)
    models = {"six_features": baseline, "six_features_time": timed, "six_features_adaptive": copy.deepcopy(baseline)}
    rows, updates = evaluate_chronologically(models, features, targets, args, device, run_dir)
    summary = summarize(rows, updates, args)
    summary["run_dir"] = str(run_dir.resolve())
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_evolution(rows, args, run_dir)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
