"""Compare the current SO3 model with a Transformer under a shared protocol."""
import json
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.mamba import SO3_MODEL_TYPES
from src.model.loss import AngularLoss, PoincareRegularizedMSE, Infidelity
from src.training.args import parse_args
from src.training.data import SParameterDataset, batch_inputs
from src.training.mamba_training import set_seed, make_optimizer, epoch_loss
from src.utils.data import REPO_ROOT, DATASET_PATHS, load_stokes
from src.utils.metrics import prediction_metrics
from src.utils.runs import create_run_directory


# Transformer Model
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out(out)
        return out

class SParameterPredictionModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.1)
        self.attn_layers = nn.ModuleList([
            SelfAttention(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, 3)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]

        for attn, norm in zip(self.attn_layers, self.norm_layers):
            residual = x
            x = attn(x)
            x = norm(x + residual)

        x = x[:, -1, :]
        return self.output(x)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_model(model, train_loader, val_loader, args, criterion, device, checkpoint_path, config):
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    best_val, stale = float("inf"), 0
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        train_loss = epoch_loss(model, train_loader, criterion, device, optimizer)
        val_loss = epoch_loss(model, val_loader, criterion, device)
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError("Non-finite comparison loss")
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save({"format_version": 1, "state_dict": model.state_dict(), "config": config,
                        "training_config": vars(args), "best_val_loss": best_val}, checkpoint_path)
        else:
            stale += 1
        scheduler.step(val_loss)
        if stale >= 10:
            break
    synchronize(device)
    elapsed = time.perf_counter() - start
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["state_dict"])
    return {"training_seconds": elapsed, "epochs_run": epoch + 1, "best_val_loss": best_val,
            "checkpoint_path": str(checkpoint_path.resolve())}


def evaluate_and_benchmark(model, test_loader, device):
    model.eval()
    preds, actuals, total_time, count = [], [], 0.0, 0
    # Warm up kernels before timing; transfers and metrics are outside forward timing.
    warmup_x, _ = next(iter(test_loader))
    warmup_x = batch_inputs(warmup_x, device)
    with torch.inference_mode():
        for _ in range(5):
            model(*warmup_x)
        synchronize(device)
        for x, y in test_loader:
            x = batch_inputs(x, device)
            synchronize(device)
            start = time.perf_counter()
            output = model(*x)
            synchronize(device)
            total_time += time.perf_counter() - start
            count += len(y)
            preds.append(output.cpu().numpy().reshape(-1, 3))
            actuals.append(y.numpy().reshape(-1, 3))
    return {**prediction_metrics(np.concatenate(preds), np.concatenate(actuals)),
            "forward_seconds": total_time, "seconds_per_window": total_time / count,
            "windows_per_second": count / total_time, "test_windows": count}


def main(argv=None):
    args = parse_args(argv)
    if args.validation_only:
        raise ValueError("Use the HPO entry point for validation-only runs")
    if args.dim % 4:
        raise ValueError("Transformer comparison requires --dim divisible by 4")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    path = args.data_path or REPO_ROOT / DATASET_PATHS[args.wavelength_range]
    features, targets = load_stokes(path)
    features, targets = features[:args.max_samples], targets[:args.max_samples]
    train_end, val_end = int(0.7 * len(features)), int(0.8 * len(features))
    splits = [slice(0, train_end), slice(train_end, val_end), slice(val_end, None)]
    run_dir = create_run_directory(args.output_dir, "comparison", args.run_id)
    reg = args.lambda_reg
    criterion = {"mse": nn.MSELoss, "infidelity": Infidelity,
                 "regmse": lambda: PoincareRegularizedMSE(0.2 if reg is None else reg),
                 "angular": lambda: AngularLoss(0.02 if reg is None else reg)}[args.loss.lower()]()
    results = {"protocol": {**vars(args), "data_path": str(path), "num_samples": len(features),
                            "device": str(device), "precision": "float32", "torch_version": str(torch.__version__)}}
    mamba_name = args.model_type.replace("Polarization", "", 1)
    for name in (mamba_name, "Transformer"):
        # Reset RNG and shuffle generator so both models see the same minibatch order.
        set_seed(args.seed)
        include_time = name == mamba_name and getattr(SO3_MODEL_TYPES[args.model_type], "requires_time", False)
        time_step = args.sample_interval or 1.0
        time_config = {"time_origin": 0.0, "time_scale": (train_end - 1) * time_step} if include_time else {}
        model = (SO3_MODEL_TYPES[args.model_type](3, args.dim, args.layers, device=device, backend=args.backend, **time_config) if name == mamba_name
                 else SParameterPredictionModel(3, args.dim, 4, args.layers, max_seq_len=args.window_size)).to(device)
        datasets = [SParameterDataset(features[s], targets[s], args.window_size, start_index=s.start or 0,
                                      include_time=include_time, time_step=time_step) for s in splits]
        loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0),
                              generator=torch.Generator().manual_seed(args.seed)) for i, ds in enumerate(datasets)]
        config = {"model_type": type(model).__name__,
                  "input_dim": 3, "d_model": args.dim, "n_layers": args.layers, "window_size": args.window_size,
                  **time_config}
        if name == mamba_name:
            config["backend"] = model.backend
            if include_time:
                config["time_units"] = "seconds" if args.sample_interval else "sample_index"
        else:
            config.update({"n_heads": 4, "dropout": 0.1, "max_seq_len": args.window_size})
        timing = train_model(model, loaders[0], loaders[1], args, criterion, device, run_dir / f"{name}.pt", config)
        metrics = evaluate_and_benchmark(model, loaders[2], device)
        results[name] = {**timing, **metrics, "parameters": sum(p.numel() for p in model.parameters()), "config": config}
        print(f"{name}: {results[name]}")
        del model
    with (run_dir / "metrics.json").open("w") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    return results


if __name__ == "__main__":
    main()
