import numpy as np


def prediction_metrics(preds, actuals):
    preds = np.asarray(preds).reshape(-1, 3)
    actuals = np.asarray(actuals).reshape(-1, 3)
    norms = np.linalg.norm(preds, axis=-1, keepdims=True)
    normed = preds / np.clip(norms, 1e-8, None)
    target_unit = actuals / np.clip(np.linalg.norm(actuals, axis=-1, keepdims=True), 1e-8, None)
    mse = float(np.mean((preds - actuals) ** 2))
    norm_mse = float(np.mean((normed - actuals) ** 2))
    return {
        "test_mse": mse,
        "test_rmse": float(np.sqrt(mse)),
        "test_mae": float(np.mean(np.abs(preds - actuals))),
        "norm_test_mse": norm_mse,
        "norm_test_rmse": float(np.sqrt(norm_mse)),
        "norm_test_mae": float(np.mean(np.abs(normed - actuals))),
        "mean_deviation": float(np.mean(np.abs(norms - 1.0))),
        "mean_fidelity": float(np.mean(0.5 * (1 + np.clip(np.sum(normed * target_unit, axis=-1), -1, 1)))),
    }
