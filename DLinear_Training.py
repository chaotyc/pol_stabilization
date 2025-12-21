import os
import scipy.io
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from typing import Tuple

# Verify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
print("Current device:", device)

# DLinear Configuration
class Config:
    def __init__(self):
        self.seq_len = 64       # Input window size (matches original)
        self.pred_len = 1       # Predict 1 step ahead (matches original target)
        self.individual = False # Shared weights across channels (can be set to True)
        self.enc_in = 3         # Number of input channels (S1, S2, S3)

# --- 2. DATA LOADING ---

# Load the 5 nm separation dataset
# Note: Assuming file path structure matches the repo or upload
data_path = "07_19_2025100k_samples_txp_1551.5_pax_1556.5_polcon_and_fiber_1Hz.mat"

# In a real run, ensure this path points to the correct uploaded file location
if not os.path.exists(data_path):
    # Fallback to look in standard upload directory if local run
    possible_paths = [
        "../../Data/basic_datasets/" + data_path,
        "Data/basic_datasets/" + data_path,
        data_path
    ]
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break

print("Loading S-parameter data from:", data_path)

try:
    mat_data = scipy.io.loadmat(data_path)
    
    # Extract S-parameter data
    s1_pax = mat_data['s1_pax'].flatten()  # Reference S-parameters (Target)
    s2_pax = mat_data['s2_pax'].flatten()
    s3_pax = mat_data['s3_pax'].flatten()

    s1_txp = mat_data['s1_txp'].flatten()  # Input S-parameters (Control)
    s2_txp = mat_data['s2_txp'].flatten()
    s3_txp = mat_data['s3_txp'].flatten()

    # Stack features (inputs) and targets (references)
    features = np.column_stack([s1_txp, s2_txp, s3_txp])  # 3 input S-parameters
    targets = np.column_stack([s1_pax, s2_pax, s3_pax])   # 3 reference S-parameters

    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")

except FileNotFoundError:
    print("Dataset file not found. Please ensure the .mat file is in the working directory.")
    # creating dummy data for code verification purposes if file is missing
    features = np.random.randn(1000, 3)
    targets = np.random.randn(1000, 3)

# --- 3. DATASET CLASS ---

class SParameterDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, window_size: int, indices: np.ndarray = None):
        # Check for NaN or infinite values
        if np.isnan(features).any() or np.isnan(targets).any():
            print("Warning: NaN values found in data!")
            valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(targets).any(axis=1))
            features = features[valid_mask]
            targets = targets[valid_mask]
            
        if np.isinf(features).any() or np.isinf(targets).any():
            print("Warning: Infinite values found in data!")
            valid_mask = ~(np.isinf(features).any(axis=1) | np.isinf(targets).any(axis=1))
            features = features[valid_mask]
            targets = targets[valid_mask]
        
        # Normalize the data
        self.features_mean = np.mean(features, axis=0)
        self.features_std = np.std(features, axis=0) + 1e-8
        self.targets_mean = np.mean(targets, axis=0)
        self.targets_std = np.std(targets, axis=0) + 1e-8
        
        self.features = (features - self.features_mean) / self.features_std
        self.targets = (targets - self.targets_mean) / self.targets_std
        
        self.window_size = window_size
        self.indices = indices if indices is not None else np.arange(len(self.features) - window_size + 1)
        self.length = len(self.indices)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        data_idx = self.indices[idx]
        window = self.features[data_idx:data_idx + self.window_size]
        target = self.targets[data_idx + self.window_size - 1]
        time_index = data_idx + self.window_size - 1
        return torch.FloatTensor(window), torch.FloatTensor(target), time_index
    
    def denormalize_predictions(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        return predictions * self.targets_std + self.targets_mean
    
    def denormalize_targets(self, targets):
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        return targets * self.targets_std + self.targets_mean

# --- 4. DLINEAR MODEL DEFINITION ---

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearModel(nn.Module):
    """
    DLinear Implementation adapted for S-Parameter Prediction
    """
    def __init__(self, configs):
        super(DLinearModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # Linear Decoder is often not used in standard DLinear simple implementation, 
                # but we keep structure if needed. For simple DLinear, we sum seasonal+trend.
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Init weights
            self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        
        # Original DLinear returns [Batch, Output length, Channel]
        # Our target is [Batch, Channel] (since Output length = 1)
        # We permute back to [Batch, 1, 3] then squeeze to [Batch, 3]
        return x.permute(0,2,1).squeeze(1)

# --- 5. INITIALIZATION AND SPLITTING ---

window_size = 64
batch_size = 64

# Create Dataset
dataset_full = SParameterDataset(features, targets, window_size)
total_length = len(dataset_full)
train_length = int(0.8 * total_length)
test_length = total_length - train_length

print(f"Dataset split:")
print(f"- Total samples: {total_length}")
print(f"- Training: {train_length}")
print(f"- Testing: {test_length}")

train_indices = np.arange(train_length)
test_indices = np.arange(train_length, total_length)

train_dataset = SParameterDataset(features, targets, window_size, train_indices)
test_dataset = SParameterDataset(features, targets, window_size, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model
configs = Config()
model = DLinearModel(configs).to(device)

print(f"Model created and moved to {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# --- 6. TRAINING LOOP ---

def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
                epochs: int, device: torch.device, lr: float = 1e-4):
    print("Model device:", next(model.parameters()).device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.MSELoss()
    
    train_losses, test_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (batch_x, batch_y, _) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Debug shape on first batch
            if i == 0 and epoch == 0:
                print(f"Batch shapes - Input: {batch_x.shape}, Target: {batch_y.shape}")
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                output = model(batch_x)
                loss = criterion(output, batch_y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, _ in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with torch.amp.autocast('cuda'):
                    output = model(batch_x)
                test_loss += criterion(output, batch_y).item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    return train_losses, test_losses

# Run Training
epochs = 200 
print(f"Starting training for {epochs} epochs...")
train_losses, test_losses = train_model(model, train_loader, test_loader, epochs, device)

# Evaluation

def evaluate_model(model, test_loader, dataset):
    model.eval()
    predictions = []
    actuals = []
    time_indices = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_indices in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            with torch.amp.autocast('cuda'):
                output = model(batch_x)
            
            # Denormalize predictions and targets
            output_denorm = dataset.denormalize_predictions(output)
            batch_y_denorm = dataset.denormalize_targets(batch_y)
            
            predictions.append(output_denorm)
            actuals.append(batch_y_denorm)
            time_indices.append(batch_indices.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    time_indices = np.concatenate(time_indices, axis=0)
    
    # Sort by time index
    sort_idx = np.argsort(time_indices)
    time_indices = time_indices[sort_idx]
    predictions = predictions[sort_idx]
    actuals = actuals[sort_idx]
    
    return predictions, actuals, time_indices

# Evaluate on test set
predictions, actuals, time_indices = evaluate_model(model, test_loader, test_dataset)

# Calculate RMSE for each S-parameter
rmse_s1 = np.sqrt(np.mean((predictions[:, 0] - actuals[:, 0])**2))
rmse_s2 = np.sqrt(np.mean((predictions[:, 1] - actuals[:, 1])**2))
rmse_s3 = np.sqrt(np.mean((predictions[:, 2] - actuals[:, 2])**2))

print(f"RMSE Results (DLinear):")
print(f"S1: {rmse_s1:.6f}")
print(f"S2: {rmse_s2:.6f}")
print(f"S3: {rmse_s3:.6f}")
print(f"Average RMSE: {(rmse_s1 + rmse_s2 + rmse_s3)/3:.6f}")

# Plot Training History
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('DLinear Training Convergence')
plt.legend()
plt.show()