import os
import json
import math
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from args import parse_args
from mamba import PolarizationMamba
from loss import AngularLoss, PoincareRegularizedMSE
from plotting import output_results
import platform
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    # Optional: forces deterministic algorithms (can slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

set_seed(42)

# Check system OS for appropriate Mamba implementation
system_os = platform.system()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
print("Current device:", device)
torch.backends.cudnn.benchmark = True # Optimizes performance for fixed input sizes

class SParameterDataset(Dataset):
    def __init__(self, features, targets, window_size, pred_len):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.window_size = window_size
        self.pred_len = pred_len 

    def __len__(self):
        return len(self.features) - self.window_size - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        
        start_idx = idx + self.window_size
        y = self.targets[start_idx : start_idx + self.pred_len]
        
        return x, y

if __name__ == '__main__':
    # Parse Arguments
    args = parse_args()
    window_size = args.window_size
    pred_len = args.pred_len
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    delta_lambda = args.wavelength_range

    lambda_reg = args.lambda_reg
    loss_constructors = {
        "mse": lambda: nn.MSELoss(),
        "regmse": lambda: PoincareRegularizedMSE(lambda_reg=lambda_reg if lambda_reg is not None else 0.2),
        "angular": lambda: AngularLoss(lambda_reg=lambda_reg if lambda_reg is not None else 0.02),
    }
    loss_type = loss_constructors[args.loss.lower()]()

    if delta_lambda == "1mm":
        path = "Datasets/07_19_2025100k_samples_txp_1551.5_pax_1552.5_polcon_and_fiber_1Hz.mat"
    elif delta_lambda == "5mm":
        path = "Datasets/03_02_2026400k_samples_txp_1551.5_pax_1556.5_polcon_and_fiber_2_1Hz.mat"
    elif delta_lambda == "10mm":
        path = "Datasets/03_02_2026400k_samples_txp_1551.5_pax_1561.5_polcon_and_fiber_2_1Hz.mat"
    elif delta_lambda == "14mm":
        path = "Datasets/03_02_2026400k_samples_txp_1551.5_pax_1565.496_polcon_and_fiber_2_1Hz.mat"
    elif delta_lambda == "-5mm":
        path = "Datasets/03_02_2026400k_samples_txp_1551.5_pax_1546.5_polcon_and_fiber_2_1Hz.mat"
    else:
        print(f"Dataset does not exist for wavelength range {delta_lambda}")
        exit(1)
        
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

    # for data testing, take first 100k subset
    MAX_SAMPLES = 100000
    features = features[:MAX_SAMPLES]
    targets = targets[:MAX_SAMPLES]

    train_end = int(0.7 * len(features))
    val_end = int(0.8 * len(features))

    train_features = features[:train_end]
    train_targets = targets[:train_end]
    val_features = features[train_end:val_end]
    val_targets = targets[train_end:val_end]
    test_features = features[val_end:]
    test_targets = targets[val_end:]

    # Prepare Data
    train_set = SParameterDataset(train_features, train_targets, window_size, pred_len)
    val_set = SParameterDataset(val_features, val_targets, window_size, pred_len)
    test_set = SParameterDataset(test_features, test_targets, window_size, pred_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize Model
    model = PolarizationMamba(input_dim=3, d_model=args.dim, n_layers=args.layers, system=system_os, pred_len=args.pred_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = loss_type

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    print("Starting Training...")

    # Training Loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = 'Results/best_model_MAMBA.pt'

    patience = 10
    static_epochs = 0
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
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                output = model(x)
                epoch_val_losses.append(criterion(output, y).item())
                
        val_loss = np.mean(epoch_val_losses)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            static_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | * Best model saved")
        else:
            static_epochs += 1
            tqdm.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | No improvement for {static_epochs} epochs")
            
            if static_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break

    run_tag = f"_{args.run_id}" if args.run_id else ""
    model_info = f"{args.dim}x{args.layers}_LR{lr}_Loss{args.loss}_PL{pred_len}_dataset{args.wavelength_range}{run_tag}"

    # Training Convergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'MAMBA Training Convergence\n({model_info})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Results/MAMBA_convergence_{model_info}.png')

    # Load best validation model for final evaluation
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print(f"Loaded best model (val loss: {best_val_loss:.6f}) for final evaluation on test set")
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds.append(output.cpu().detach().numpy())
            actuals.append(y.cpu().detach().numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    # Evaluation Plotting
    # Number of points plotted (starting at end of training data)
    N_PLOT = 1000

    output_results(preds, actuals, val_end, window_size, model_info, args, pred_len, N_PLOT)

    # Save test metrics for sweep analysis
    test_mse = float(np.mean((preds - actuals) ** 2))
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(np.mean(np.abs(preds - actuals)))
    pred_norms = np.linalg.norm(preds.reshape(-1, 3), axis=1)
    mean_deviation = float(np.mean(np.abs(pred_norms - 1.0)))


    # Force normalization of predictions to poincare sphere and calculate metrics
    preds_flat = preds.reshape(-1, 3)
    actuals_flat = actuals.reshape(-1, 3)
    norms = np.linalg.norm(preds_flat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    preds_normed = (preds_flat / norms).reshape(preds.shape)
    norm_mse = float(np.mean((preds_normed - actuals) ** 2))
    norm_rmse = float(np.sqrt(norm_mse))
    norm_mae = float(np.mean(np.abs(preds_normed - actuals)))

    metrics = {
        'wavelength_range': delta_lambda,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'norm_test_mse': norm_mse,
        'norm_test_rmse': norm_rmse,
        'norm_test_mae': norm_mae,
        'mean_deviation': mean_deviation,
        'best_val_loss': float(best_val_loss),
        'model_info': model_info,
    }
    metrics_path = f'Results/MAMBA_test_results_{delta_lambda}{run_tag}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Test metrics saved to {metrics_path}")