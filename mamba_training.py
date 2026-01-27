import os
import math
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from args import parse_args
from mamba import MambaBlock, PolarizationMamba
from loss import AngularLoss, PoincareRegularizedMSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
print("Current device:", device)
torch.backends.cudnn.benchmark = True # Optimizes performance for fixed input sizes

# Parse Arguments
args = parse_args()
window_size = args.window_size
pred_len = args.pred_len
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
loss_type = {"MSE": nn.MSELoss(),
             "RegMSE": PoincareRegularizedMSE(lambda_reg=0.1),
             "Angular": AngularLoss()}[args.loss.lower()]

path = "Datasets/07_19_2025100k_samples_txp_1551.5_pax_1556.5_polcon_and_fiber_1Hz.mat"
if not os.path.exists(path):
    print(f"File not found at {path}")
    exit(1)
else:
    print(f"Loading data from {path}")
    mat_data = scipy.io.loadmat(path)

# Extract Data
s1_pax = mat_data['s1_pax'].flatten()
s2_pax = mat_data['s2_pax'].flatten()
s3_pax = mat_data['s3_pax'].flatten()
s1_txp = mat_data['s1_txp'].flatten()
s2_txp = mat_data['s2_txp'].flatten()
s3_txp = mat_data['s3_txp'].flatten()

features = np.column_stack([s1_txp, s2_txp, s3_txp])
targets = np.column_stack([s1_pax, s2_pax, s3_pax])

split_idx = int(0.8 * len(features))

train_features = features[:split_idx]
train_targets = targets[:split_idx]
test_features = features[split_idx:]
test_targets = targets[split_idx:]

# Compute normalization statistics from training data
f_mean = torch.FloatTensor(train_features.mean(axis=0))
f_std = torch.FloatTensor(train_features.std(axis=0) + 1e-6)
t_mean = torch.FloatTensor(train_targets.mean(axis=0))
t_std = torch.FloatTensor(train_targets.std(axis=0) + 1e-6)

class SParameterDataset(Dataset):
    def __init__(self, features, targets, window_size, f_mean, f_std, t_mean, t_std):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
        # Store stats for denormalization later
        self.t_mean = t_mean
        self.t_std = t_std
        
        # Normalize using the PASSED statistics
        self.features = (self.features - f_mean) / f_std
        self.targets = (self.targets - t_mean) / t_std
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        return self.features[idx : idx + self.window_size], self.targets[idx + self.window_size - 1]

    def denorm(self, y):
        if isinstance(y, torch.Tensor): y = y.cpu().detach().numpy()
        return y * self.t_std.numpy() + self.t_mean.numpy()

# Prepare Data
train_set = SParameterDataset(train_features, train_targets, window_size, f_mean, f_std, t_mean, t_std)
test_set = SParameterDataset(test_features, test_targets, window_size, f_mean, f_std, t_mean, t_std)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Initialize Model
model = PolarizationMamba(input_dim=3, d_model=args.dim, n_layers=args.layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
criterion = loss_type

print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
print("Starting Training...")

# Training Loop
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    batch_losses = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        
        # Update the progress bar with the current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            output = model(x)
            val_losses.append(criterion(output, y).item())
            
    test_loss = np.mean(val_losses)
    test_losses.append(test_loss)
    
    tqdm.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {test_loss:.6f}")

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('MAMBA Training Convergence')
plt.legend()
plt.savefig('MAMBA_training_convergence.png')

# Final Evaluation
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        output = model(x)
        preds.append(train_set.denorm(output))
        actuals.append(train_set.denorm(y))

preds = np.concatenate(preds)
actuals = np.concatenate(actuals)

# Evaluation Plotting
# Number of points plotted (starting at end of training data)
N_PLOT = 200
# Slice the first N_PLOT samples from the test results
preds_slice = preds[:N_PLOT]
actuals_slice = actuals[:N_PLOT]
errors_slice = np.abs(preds_slice - actuals_slice)

# Create the correct time indices for the x-axis
start_time_index = split_idx 
time_indices = range(start_time_index + window_size, start_time_index + window_size + N_PLOT)

plt.figure(figsize=(15, 10))
params = ['S1', 'S2', 'S3']

for i in range(3):
    # Predictions vs Actuals
    plt.subplot(3, 2, (i*2)+1)
    plt.plot(time_indices, actuals_slice[:, i], label='Actual', color='blue', linewidth=1.5)
    plt.plot(time_indices, preds_slice[:, i], label='Predicted', color='red', linestyle='--', linewidth=1.5)
    plt.title(f'{params[i]} Parameter Time Series')
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
plt.savefig('MAMBA_predictions.png')

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
plt.title('Physical Consistency Check: Magnitude of Predicted Stokes Vectors')
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