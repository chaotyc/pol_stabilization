import numpy as np
import matplotlib.pyplot as plt

def output_results(preds, actuals, split_idx, window_size, model_info, args, pred_len, n_plot=500):
    if pred_len > 1:
        last_step_idx = pred_len - 1
        # Slices for plotting
        truth_slice = actuals[:n_plot + pred_len, 0, :]
        preds_first = preds[:n_plot, 0, :]
        preds_last = preds[:n_plot, -1, :]

        # Time indices
        start_t = split_idx + window_size
        time_first = range(start_t, start_t + n_plot)
        time_last = range(start_t + last_step_idx, start_t + last_step_idx + n_plot)
        time_truth = range(start_t, start_t + n_plot + pred_len)

        plt.figure(figsize=(15, 12))
        params = ['S1', 'S2', 'S3']

        for i in range(3):
            # Time Series Comparison
            plt.subplot(3, 2, (i*2)+1)
            plt.plot(time_truth[:len(truth_slice)], truth_slice[:, i], label='Actual', color='black', alpha=0.3, linewidth=3)
            plt.plot(time_first, preds_first[:, i], label='Pred (Step 1)', color='blue', linestyle='--', linewidth=1)
            plt.plot(time_last, preds_last[:, i], label=f'Pred (Step {pred_len})', color='red', linestyle='-', linewidth=1)
            plt.title(f'{params[i]} Time Series ({model_info})')
            plt.xlabel('Time Index')
            plt.ylabel('Value')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.5)

            # Error Comparison
            plt.subplot(3, 2, (i*2)+2)
            error_first = np.abs(preds[:n_plot, 0, i] - actuals[:n_plot, 0, i])
            error_last = np.abs(preds[:n_plot, -1, i] - actuals[:n_plot, -1, i])
            plt.plot(time_first, error_first, label='Error (Step 1)', color='blue', alpha=0.6)
            plt.plot(time_first, error_last, label=f'Error (Step {pred_len})', color='red', alpha=0.6)
            plt.title(f'{params[i]} Absolute Error')
            plt.xlabel('Time Index')
            plt.legend()
            plt.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'MAMBA_predictions_{model_info}.png')

        # Calculate Squared Errors
        squared_errors = (preds - actuals) ** 2

        # Calculate MSE per step
        # Average over axis 0 (samples) and axis 2 (features S1,S2,S3)
        mse_per_step = np.mean(squared_errors, axis=(0, 2))

        # Calculate RMSE per step
        rmse_per_step = np.sqrt(mse_per_step)

        print(f"\nStatistics over {args.pred_len}-step Prediction Horizon:")

        print("\nMean MSE by step:")
        print(mse_per_step)

        print("\nMean RMSE by step:")
        print(rmse_per_step)

        # Plot the error growth over time
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, args.pred_len + 1), rmse_per_step, marker='o', linestyle='-')
        plt.title(f'Error Growth over Prediction Horizon ({args.pred_len} steps)')
        plt.xlabel('Prediction Step')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.savefig('error_over_preds.png')

        # calculate deviation from norm
        # Plot L2 Norms of predictions to determine if they deviate from 1
        # Calculate L2 Norms of the predictions
        predicted_norms = np.linalg.norm(preds_first, axis=1)

        # Calculate Deviation from poincare unit sphere
        deviation_from_unity = np.abs(predicted_norms - 1)

        # Plot L2 Norm vs Time
        plt.figure(figsize=(12, 6))
        plt.plot(time_first, predicted_norms, label='Predicted L2 Norm', color='green', linewidth=1.5)
        plt.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Unit Sphere (Ideal = 1.0)')
        plt.title(f'Physical Consistency Check ({model_info})')
        plt.xlabel('Time Index')
        plt.ylabel('Vector Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig('MAMBA_s_parameter_norms.png')

        # Print Deviation Statistics
        mean_dev = np.mean(deviation_from_unity)
        max_dev = np.max(deviation_from_unity)

        print("Physical Validity Statistics:")
        print(f"Mean Deviation from Unit Norm: {mean_dev:.6f}")
        print(f"Max Deviation from Unit Norm:  {max_dev:.6f}")

    else:
        # For single-step prediction, we can directly compare the predictions to actuals and plot errors
        preds_slice = preds[:n_plot, 0, :]
        actuals_slice = actuals[:n_plot, 0, :]
        errors_slice = np.abs(preds_slice - actuals_slice)

        # Create the correct time indices for the x-axis
        start_time_index = split_idx 
        time_indices = range(start_time_index + window_size, start_time_index + window_size + n_plot)

        plt.figure(figsize=(15, 10))
        params = ['S1', 'S2', 'S3']

        for i in range(3):
            # Predictions vs Actuals
            plt.subplot(3, 2, (i*2)+1)
            plt.plot(time_indices, actuals_slice[:, i], label='Actual', color='blue', linewidth=1.5)
            plt.plot(time_indices, preds_slice[:, i], label='Predicted', color='red', linestyle='--', linewidth=1.5)
            plt.title(f'{params[i]} Parameter Time Series ({model_info})')
            plt.xlabel('Time Index')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.5)

            # Error Plot
            plt.subplot(3, 2, (i*2)+2)
            plt.plot(time_indices, errors_slice[:, i], label='Abs Error', color='purple', alpha=0.8)
            plt.title(f'{params[i]} Absolute Error')
            plt.xlabel('Time Index')
            plt.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'MAMBA_predictions_{model_info}.png')

        # Print Statistics for the slice
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

        # Plot L2 Norms of predictions to determine if they deviate from 1
        # Calculate L2 Norms of the predictions
        predicted_norms = np.linalg.norm(preds_slice, axis=1)

        # Calculate Deviation from poincare unit sphere
        deviation_from_unity = np.abs(predicted_norms - 1)

        # Plot L2 Norm vs Time
        plt.figure(figsize=(12, 6))
        plt.plot(time_indices, predicted_norms, label='Predicted L2 Norm', color='green', linewidth=1.5)
        plt.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Unit Sphere (Ideal = 1.0)')
        plt.title(f'Physical Consistency Check ({model_info})')
        plt.xlabel('Time Index')
        plt.ylabel('Vector Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig('MAMBA_s_parameter_norms.png')

        # Print Deviation Statistics
        mean_dev = np.mean(deviation_from_unity)
        max_dev = np.max(deviation_from_unity)

        print("Physical Validity Statistics:")
        print(f"Mean Deviation from Unit Norm: {mean_dev:.6f}")
        print(f"Max Deviation from Unit Norm:  {max_dev:.6f}")