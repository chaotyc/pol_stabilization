import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Config
path = "Datasets/07_19_2025100k_samples_txp_1551.5_pax_1556.5_polcon_and_fiber_1Hz.mat"
n_plot = 200
split_ratio = 0.8

if not os.path.exists(path):
    print(f"File not found at {path}")
    exit(1)

# Load Data
print(f"Loading data from {path}")
mat_data = scipy.io.loadmat(path)

s1_pax = mat_data['s1_pax'].flatten()
s2_pax = mat_data['s2_pax'].flatten()
s3_pax = mat_data['s3_pax'].flatten()

targets = np.column_stack([s1_pax, s2_pax, s3_pax])

# Split Data
split_idx = int(split_ratio * len(targets))
test_targets = targets[split_idx:]

print(f"Test Set Size: {len(test_targets)}")

# Naive Forecast (Persistence)
# Assumption: x[t] = x[t-1]
actuals = test_targets[1:]
naive_preds = test_targets[:-1]

# Evaluation Metrics
mse = mean_squared_error(actuals, naive_preds)
mae = mean_absolute_error(actuals, naive_preds)
rmse = np.sqrt(mse)

print(f"Overall MSE:  {mse:.6f}")
print(f"Overall MAE:  {mae:.6f}")
print(f"Overall RMSE: {rmse:.6f}")

# Per-parameter stats
params = ['S1', 'S2', 'S3']
errors = np.abs(naive_preds - actuals)

for i in range(3):
    p_mae = np.mean(errors[:, i])
    p_rmse = np.sqrt(np.mean(errors[:, i]**2))
    print(f"{params[i]} - MAE: {p_mae:.5f}, RMSE: {p_rmse:.5f}")

# Plotting Results
preds_slice = naive_preds[:n_plot]
actuals_slice = actuals[:n_plot]
errors_slice = errors[:n_plot]
time_indices = range(n_plot)

plt.figure(figsize=(15, 10))

for i in range(3):
    # Predictions vs Actuals
    plt.subplot(3, 2, (i*2)+1)
    plt.plot(time_indices, actuals_slice[:, i], label='Actual', color='blue', linewidth=1.5)
    plt.plot(time_indices, preds_slice[:, i], label='Naive', color='green', linestyle='--', linewidth=1.5)
    plt.title(f'{params[i]} Naive Forecast')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Error Plot
    plt.subplot(3, 2, (i*2)+2)
    plt.plot(time_indices, errors_slice[:, i], label='Abs Error', color='gray', alpha=0.8)
    plt.title(f'{params[i]} Error')
    plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig('naive_baseline_results.png')
print("Plot saved to naive_baseline_results.png")