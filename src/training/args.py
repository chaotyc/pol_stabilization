import argparse

from src.model.mamba import SO3_MODEL_TYPES

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mamba Training Script Arguments")

    parser.add_argument(
        "--model-type",
        choices=SO3_MODEL_TYPES,
        default="PolarizationMambaSO3",
        help="SO3 input model: differences, coordinates plus differences, or CombinedTime with a timestamp",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Size of the input window for the model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--loss",
        type=str.lower,
        choices=["mse", "regmse", "angular", "infidelity"],
        default="infidelity",
        help="Loss function to use (MSE, RegMSE, Angular, Infidelity)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=32,
        help="Dimensionality of the model",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help="Number of layers in the model",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 penalty) for AdamW optimizer",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=None,
        help="Regularization strength for RegMSE/Angular loss (default: 0.2 for RegMSE, 0.02 for Angular)",
    )
    parser.add_argument(
        "--wavelength-range",
        type=str,
        default="loop_5mm",
        help="Wavelength datasets to use (synthetic_1mm, synthetic_5mm, synthetic_10mm, synthetic_14mm, loop_1mm, loop_5mm, loop_10mm, loop_14mm)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique run identifier for output files (used for hyperparameter tuning)",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.2,
        help="Factor to reduce LR by on plateau (default: 0.2)",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Epochs with no val improvement before reducing LR (default: 5)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-7,
        help="Minimum LR floor for scheduler (default: 1e-7)",
    )

    parser.add_argument("--output-dir", default="results", help="Root directory for isolated run artifacts")
    parser.add_argument("--data-path", default=None, help="Override the selected dataset path")
    parser.add_argument("--max-samples", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--backend", choices=["auto", "pytorch", "cuda"], default="auto")
    parser.add_argument("--validation-only", action="store_true", help="Skip test evaluation during HPO")
    parser.add_argument("--sample-interval", type=float, default=None, help="Seconds per sample; plots use sample indices if omitted")
    args = parser.parse_args(argv)
    if args.window_size < 2:
        parser.error("--window-size must be at least 2 for the SO3 model")
    for name in ("epochs", "batch_size", "dim", "layers", "max_samples", "lr"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.sample_interval is not None and args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")
    return args
