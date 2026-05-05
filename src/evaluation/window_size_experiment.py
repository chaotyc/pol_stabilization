import os
import subprocess
import json
import matplotlib.pyplot as plt
import sys

# Experiment Configuration
window_sizes = [1, 2, 4, 16, 64, 256]
dim = 32
wavelength = "loop_5mm" # Target loop 5mm dataset
epochs = 20        # Adjust if you need longer/shorter training for the experiment

results_mse = []
results_fidelity = []

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

print(f"Testing window sizes: {window_sizes} with Model Dim: {dim}")

for w in window_sizes:
    print(f"\n{'='*50}")
    print(f"Running training for Window Size: {w}")
    print(f"{'='*50}")
    
    # Construct the command to run your training script
    cmd = [
        sys.executable, "src/training/mamba_training.py",
        "--window-size", str(w),
        "--dim", str(dim),
        "--wavelength-range", wavelength,
        "--epochs", str(epochs)
    ]
    
    # Execute the training run
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training for window size {w}: {e}")
        break

    # The training script outputs to results/MAMBA_test_results_{wavelength}.json 
    result_file = f"results/MAMBA_test_results_{wavelength}.json"
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            # Extract MSE and Fidelity
            mse = data.get("test_mse", None)
            fidelity = data.get("mean_fidelity", None)
            
            results_mse.append(mse)
            results_fidelity.append(fidelity)
            
            print(f"Completed Window {w} -> MSE: {mse:.4f}, Fidelity: {fidelity:.4f}")
            
        # Rename the result file to avoid overwriting it in the next loop iteration
        archive_name = f"results/MAMBA_test_results_{wavelength}_dim{dim}_w{w}.json"
        if os.path.exists(archive_name):
            os.remove(archive_name)
        os.rename(result_file, archive_name)
    else:
        print(f"Warning: Expected results file {result_file} not found!")
        results_mse.append(None)
        results_fidelity.append(None)

# Plot Results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot 1: Mean Squared Error vs Window Size (Left Y-axis)
color1 = 'tab:red'
ax1.set_xlabel('Window Size (Log Scale)', fontsize=12)
ax1.set_ylabel('Test MSE', fontsize=12, color=color1)
line1 = ax1.plot(window_sizes, results_mse, marker='o', color=color1, linestyle='-', linewidth=2, markersize=8, label='Test MSE')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log', base=2)
ax1.set_xticks(window_sizes)
ax1.set_xticklabels(window_sizes)
ax1.grid(True, alpha=0.5, linestyle='--')

# Plot 2: Fidelity (Accuracy) vs Window Size (Right Y-axis)
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Mean Fidelity', fontsize=12, color=color2)
line2 = ax2.plot(window_sizes, results_fidelity, marker='s', color=color2, linestyle='-', linewidth=2, markersize=8, label='Mean Fidelity')
ax2.tick_params(axis='y', labelcolor=color2)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Impact of Memory Horizon on MSE and Fidelity', fontsize=14)

# Add a text box with experiment parameters
param_text = f"Dataset: {wavelength}\nModel Dim: {dim}\nEpochs: {epochs}"
plt.figtext(0.5, -0.05, param_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

plt.tight_layout()
output_plot = 'results/window_size_impact_results.png'
plt.savefig(output_plot, bbox_inches="tight", dpi=300)
print(f"Experiment complete! Graph saved to {output_plot}")
