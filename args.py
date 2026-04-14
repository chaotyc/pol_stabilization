import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Mamba Training Script Arguments")

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
        type=str,
        default="MSE",
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
        default=16,
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
        default="5mm",
        help="Wavelength difference for the dataset (e.g., '1mm', '5mm', '10mm', '14mm', '-5mm')",
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
        default=0.5,
        help="Factor to reduce LR by on plateau (default: 0.5)",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="Epochs with no val improvement before reducing LR (default: 3)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-7,
        help="Minimum LR floor for scheduler (default: 1e-7)",
    )

    return parser.parse_args()