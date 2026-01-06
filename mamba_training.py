import os
import math
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
print("Current device:", device)
torch.backends.cudnn.benchmark = True # Optimizes performance for fixed input sizes

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize A parameter
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: [Batch, Seq_Len, D_Model]
        batch_size, seq_len, _ = x.shape
        
        # Projections
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # Convolution
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len] # Causal Crop
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.act(x_branch)

        # 3. SSM Parameters
        x_dbl = self.x_proj(x_branch)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)) # Positive time-steps

        # Selective Scan        
        A = -torch.exp(self.A_log) # [D_inner, D_state]
        
        # Pre-calculate exp(A * dt) for all steps at once (Vectorized)
        # Shape: [Batch, Seq, Inner, State]
        dA = torch.exp(A * dt.unsqueeze(-1)) 
        
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        y = []
        
        # This loop runs on CPU, but dispatches fast kernels to GPU
        for t in range(seq_len):
            # Load current step values
            dt_t = dt[:, t, :].unsqueeze(-1)
            dA_t = dA[:, t, :, :]
            B_t = B[:, t, :].unsqueeze(1)
            C_t = C[:, t, :].unsqueeze(1)
            x_t = x_branch[:, t, :].unsqueeze(-1)
            
            # Update State: h = dA * h + B * x
            # (Note: This is the simplified discrete form)
            h = h * dA_t + (x_t * dt_t) * B_t
            
            # Output: y = C * h
            y_t = torch.sum(h * C_t, dim=-1)
            y.append(y_t)
            
        y = torch.stack(y, dim=1) # Stack results [Batch, Seq, Inner]
        
        # Add skip connection
        y = y + x_branch * self.D
        
        # Output
        out = y * self.act(z_branch)
        return self.out_proj(out)

class PolarizationMamba(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2) 
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 3) 

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x) + x
        x = self.norm_f(x)
        return self.head(x[:, -1, :])

path = "Datasets/07_19_2025100k_samples_txp_1551.5_pax_1556.5_polcon_and_fiber_1Hz.mat"
if not os.path.exists(path):
    print(f"File not found at {path}")
    exit(1)

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

class SParameterDataset(Dataset):
    def __init__(self, features, targets, window_size):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
        # Normalize (Important for convergence)
        self.f_mean, self.f_std = self.features.mean(0), self.features.std(0) + 1e-6
        self.t_mean, self.t_std = self.targets.mean(0), self.targets.std(0) + 1e-6
        
        self.features = (self.features - self.f_mean) / self.f_std
        self.targets = (self.targets - self.t_mean) / self.t_std
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        return self.features[idx : idx + self.window_size], self.targets[idx + self.window_size - 1]

    def denorm(self, y):
        if isinstance(y, torch.Tensor): y = y.cpu().detach().numpy()
        return y * self.t_std.numpy() + self.t_mean.numpy()

# Config
window_size = 64
batch_size = 64
epochs = 3

# Prepare Data
dataset = SParameterDataset(features, targets, window_size)
total_len = len(dataset)
split_idx = int(0.8 * total_len)

# Create sequential splits (First 80% for train, Last 20% for test)
train_set = torch.utils.data.Subset(dataset, range(0, split_idx))
test_set = torch.utils.data.Subset(dataset, range(split_idx, total_len))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Initialize Model
model = PolarizationMamba(input_dim=3, d_model=64, n_layers=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
criterion = nn.MSELoss()

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
plt.title('DLinear Training Convergence')
plt.legend()
plt.show()

# Final Evaluation
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        output = model(x)
        preds.append(dataset.denorm(output))
        actuals.append(dataset.denorm(y))

preds = np.concatenate(preds)
actuals = np.concatenate(actuals)

# Evaluation Plotting
# Number of points plotted (starting at end of training data)
N_PLOT = 1000
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
plt.savefig('s_parameter_predictions.png')

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
plt.savefig('s_parameter_norms.png')

# Print Deviation Statistics
mean_dev = np.mean(deviation_from_unity)
max_dev = np.max(deviation_from_unity)

print("Physical Validity Statistics:")
print(f"Mean Deviation from Unit Norm: {mean_dev:.6f}")
print(f"Max Deviation from Unit Norm:  {max_dev:.6f}")