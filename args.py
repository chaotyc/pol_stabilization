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
        "--pred-len",
        type=int,
        default=1,
        help="Length of the prediction output",
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
        help="Loss function to use (MSE, RegMSE, Angular)",
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
        default=2,
        help="Number of layers in the model",
    )

    return parser.parse_args()