import os
import subprocess
import json
import matplotlib.pyplot as plt
import sys

# Experiment Configuration
# We exclude synthetic_-5mm as requested
datasets = ["synthetic_1mm", "synthetic_5mm", "synthetic_10mm", "synthetic_14mm"]
delta_lambdas = [1, 5, 10, 14] # Corresponding delta lambda values in nm

window_size = 256
dim = 32
epochs = 50

results_mse = []
results_fidelity = []

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

print(f"Testing Delta Lambdas: {delta_lambdas}nm with Model Dim: {dim}")

for ds, dl in zip(datasets, delta_lambdas):
    print(f"\n{'='*50}")
    print(f"Running training for Dataset: {ds} (Delta Lambda: {dl}nm)")
    print(f"{'='*50}")
    
    # Construct the command to run your training script
    cmd = [
        sys.executable, "src/training/mamba_training.py",
        "--window-size", str(window_size),
        "--dim", str(dim),
        "--wavelength-range", ds,
        "--epochs", str(epochs)
    ]
    
    # Execute the training run
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training for dataset {ds}: {e}")
        break

    # The training script outputs to results/MAMBA_test_results_{wavelength_range}.json 
    result_file = f"results/MAMBA_test_results_{ds}.json"
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            # Extract MSE and Fidelity
            mse = data.get("test_mse", None)
            fidelity = data.get("mean_fidelity", None)
            
            results_mse.append(mse)
            results_fidelity.append(fidelity)
            
            print(f"Completed {ds} -> MSE: {mse:.4f}, Fidelity: {fidelity:.4f}")
            
        # Rename the result file to avoid overwriting it in the next loop iteration
        archive_name = f"results/MAMBA_test_results_{ds}_dim{dim}_w{window_size}.json"
        if os.path.exists(archive_name):
            os.remove(archive_name)
        os.rename(result_file, archive_name)
    else:
        print(f"Warning: Expected results file {result_file} not found!")
        results_mse.append(None)
        results_fidelity.append(None)

# Plot Results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot 1: Mean Squared Error vs Delta Lambda (Left Y-axis)
color1 = 'tab:red'
ax1.set_xlabel('Delta Lambda (nm)', fontsize=12)
ax1.set_ylabel('Test MSE', fontsize=12, color=color1)
line1 = ax1.plot(delta_lambdas, results_mse, marker='o', color=color1, linestyle='-', linewidth=2, markersize=8, label='Test MSE')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(delta_lambdas)
ax1.grid(True, alpha=0.5, linestyle='--')

# Plot 2: Fidelity (Accuracy) vs Delta Lambda (Right Y-axis)
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Mean Fidelity', fontsize=12, color=color2)
line2 = ax2.plot(delta_lambdas, results_fidelity, marker='s', color=color2, linestyle='-', linewidth=2, markersize=8, label='Mean Fidelity')
ax2.tick_params(axis='y', labelcolor=color2)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Impact of Delta Lambda on MSE and Fidelity', fontsize=14)

# Add a text box with experiment parameters
param_text = f"Window Size: {window_size}\nModel Dim: {dim}\nEpochs: {epochs}"
plt.figtext(0.5, -0.05, param_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

plt.tight_layout()
output_plot = 'results/delta_lambda_impact_results.png'
plt.savefig(output_plot, bbox_inches="tight", dpi=300)
print(f"Experiment complete! Graph saved to {output_plot}")