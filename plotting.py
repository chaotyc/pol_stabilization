import numpy as np
import matplotlib.pyplot as plt

def output_results(preds, actuals, split_idx, window_size, model_info, n_plot=500):
    preds_slice = preds[:n_plot, 0, :]
    actuals_slice = actuals[:n_plot, 0, :]
    errors_slice = np.abs(preds_slice - actuals_slice)

    start_time_index = split_idx 
    time_indices = range(start_time_index + window_size, start_time_index + window_size + n_plot)

    plt.figure(figsize=(15, 10))
    params = ['S1', 'S2', 'S3']

    for i in range(3):
        plt.subplot(3, 2, (i*2)+1)
        plt.plot(time_indices, actuals_slice[:, i], label='Actual', color='blue', linewidth=1.5)
        plt.plot(time_indices, preds_slice[:, i], label='Predicted', color='red', linestyle='--', linewidth=1.5)
        plt.title(f'{params[i]} Parameter Time Series ({model_info})')
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.5)

        plt.subplot(3, 2, (i*2)+2)
        plt.plot(time_indices, errors_slice[:, i], label='Abs Error', color='purple', alpha=0.8)
        plt.title(f'{params[i]} Absolute Error')
        plt.xlabel('Time Index')
        plt.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'Results/MAMBA_predictions_{model_info}.png')

    print("\n Statistics:")
    avg_mae = 0
    avg_rmse = 0
    for i in range(3):
        mae = np.mean(errors_slice[:, i])
        rmse = np.sqrt(np.mean(errors_slice[:, i]**2))
        avg_mae += mae
        avg_rmse += rmse
        print(f"{params[i]} - MAE: {mae:.5f}, RMSE: {rmse:.5f}")
    avg_mae /= 3
    avg_rmse /= 3
    print(f"Mean - MAE: {avg_mae:.5f}, RMSE: {avg_rmse:.5f}")

    predicted_norms = np.linalg.norm(preds_slice, axis=1)
    deviation_from_unity = np.abs(predicted_norms - 1)

    plt.figure(figsize=(12, 6))
    plt.plot(time_indices, predicted_norms, label='Predicted L2 Norm', color='green', linewidth=1.5)
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Unit Sphere (Ideal = 1.0)')
    plt.title(f'Physical Consistency Check ({model_info})')
    plt.xlabel('Time Index')
    plt.ylabel('Vector Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'Results/MAMBA_s_parameter_norms.png')

    mean_dev = np.mean(deviation_from_unity)
    max_dev = np.max(deviation_from_unity)

    print("Physical Validity Statistics:")
    print(f"Mean Deviation from Unit Norm: {mean_dev:.6f}")
    print(f"Max Deviation from Unit Norm:  {max_dev:.6f}")
