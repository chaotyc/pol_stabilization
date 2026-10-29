"""Measure per-window prediction latency on available devices."""
import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.inference.inference import MODEL_TYPES, PolarizationPredictor
from src.utils.data import load_stokes


def run_benchmark(predictor, features, window_size, max_windows=1000, *, start_index=0, sample_interval=None):
    n_windows = min(max_windows, len(features) - window_size)
    if n_windows < 1:
        raise ValueError("Benchmark data must contain more samples than the window size")
    needs_time = getattr(predictor.model, "requires_time", False)
    seconds = predictor.config.get("time_units") == "seconds"
    if needs_time and seconds and (sample_interval is None or not np.isfinite(sample_interval) or sample_interval <= 0):
        raise ValueError("This model uses seconds; supply --sample-interval from its training run")

    def predict_at(index):
        kwargs = {}
        if needs_time:
            kwargs["target_time"] = (start_index + index + window_size) * (sample_interval if seconds else 1.0)
        return predictor.predict(features[index:index + window_size], **kwargs)

    for _ in range(5):
        predict_at(0)
    times = []
    for i in tqdm(range(n_windows), desc=str(predictor.device)):
        if predictor.device.type == "cuda":
            torch.cuda.synchronize(predictor.device)
        start = time.perf_counter()
        predict_at(i)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize(predictor.device)
        times.append(time.perf_counter() - start)
    print(f"{predictor.device}: median={np.median(times):.6f}s, mean={np.mean(times):.6f}s, std={np.std(times):.6f}s")
    return times


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("data_path")
    parser.add_argument("--window-size", type=int, help="Original window size, required for legacy checkpoints")
    parser.add_argument("--model-type", choices=MODEL_TYPES, help="Original architecture for legacy checkpoints")
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0, help="Global sample index of the first row in data_path")
    parser.add_argument("--sample-interval", type=float, help="Seconds per sample for time models trained in seconds")
    parser.add_argument("--output", default="results/inference_time.png")
    args = parser.parse_args(argv)
    legacy = {"window_size": args.window_size, "model_type": args.model_type} if args.window_size and args.model_type else None
    features, _ = load_stokes(args.data_path)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for device in devices:
        predictor = PolarizationPredictor(args.model_path, device=device, legacy_config=legacy)
        times = run_benchmark(predictor, features, predictor.window_size, args.max_windows,
                              start_index=args.start_index, sample_interval=args.sample_interval)
        axes[0].plot(times, ".", alpha=0.5, label=device)
        axes[1].hist(times, bins=50, alpha=0.5, label=device)
    axes[0].set(xlabel="Prediction", ylabel="Latency (s)")
    axes[1].set(xlabel="Latency (s)", ylabel="Count")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
