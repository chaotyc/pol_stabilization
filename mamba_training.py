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
from plotting import output_results

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
loss_type = {"mse": nn.MSELoss(),
             "regmse": PoincareRegularizedMSE(lambda_reg=0.1),
             "angular": AngularLoss()}[args.loss.lower()]

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
    def __init__(self, features, targets, window_size, pred_len, f_mean, f_std, t_mean, t_std):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.t_mean = t_mean
        self.t_std = t_std
        self.features = (self.features - f_mean) / f_std
        self.targets = (self.targets - t_mean) / t_std
        self.window_size = window_size
        self.pred_len = pred_len 

    def __len__(self):
        return len(self.features) - self.window_size - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        
        start_idx = idx + self.window_size
        y = self.targets[start_idx : start_idx + self.pred_len]
        
        return x, y
    
    def denorm(self, y):
        if isinstance(y, torch.Tensor): y = y.cpu().detach().numpy()
        return y * self.t_std.numpy() + self.t_mean.numpy()

# Prepare Data
train_set = SParameterDataset(train_features, train_targets, window_size, pred_len,f_mean, f_std, t_mean, t_std)
test_set = SParameterDataset(test_features, test_targets, window_size, pred_len, f_mean, f_std, t_mean, t_std)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Initialize Model
model = PolarizationMamba(input_dim=3, d_model=args.dim, n_layers=args.layers, pred_len=args.pred_len).to(device)
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

model_info = f"{args.dim}x{args.layers}_LR{lr}_Loss{args.loss}_PL{pred_len}"

# Training Convergence Plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'MAMBA Training Convergence\n({model_info})')
plt.legend()
plt.grid(True)
plt.savefig(f'MAMBA_convergence_{model_info}.png')

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
N_PLOT = 500

output_results(preds, actuals, split_idx, window_size, model_info, args, pred_len, n_plot=500)