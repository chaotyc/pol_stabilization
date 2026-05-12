import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import time
import json
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import platform
import random

# Imports from Mamba environment (assuming they exist in the project structure)
from src.model.mamba import PolarizationMamba
from src.model.loss import Infidelity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
system_os = platform.system()

# Transformer Model
class FlashAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = 1.0 / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out(out)
        return out

class SParameterPredictionModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.1)
        self.attn_layers = nn.ModuleList([
            FlashAttention(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, 3)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        for attn, norm in zip(self.attn_layers, self.norm_layers):
            residual = x
            x = attn(x)
            x = norm(x + residual)
        
        x = x[:, -1, :]
        return self.output(x)

# DataLoader
class SParameterDataset(Dataset):
    def __init__(self, features, targets, window_size):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size].unsqueeze(0)
        return x, y.squeeze(0)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

# Training Loop
def train_mamba(model, train_loader, val_loader, epochs, lr, device):
    print("\n--- Starting Mamba Training ---")
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.ndim <= 1 or getattr(param, "_no_weight_decay", False) or "A_log" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": 1e-4}, 
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.MSELoss() 

    best_val_loss = float('inf')
    best_model_path = 'results/temp_best_mamba.pt'
    os.makedirs('results', exist_ok=True)

    patience = 5
    static_epochs = 0

    start_time = time.perf_counter()  # Start Training Timer

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Mamba Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(x).view(y.size())
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                output = model(x).view(y.size())
                val_loss += criterion(output, y).item()
        val_loss /= len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            static_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | * Best model saved")
        else:
            static_epochs += 1
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | No improvement for {static_epochs} epochs")
            if static_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break
            
        scheduler.step(val_loss)

    end_time = time.perf_counter()  # End Training Timer
    total_train_time = end_time - start_time

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model, total_train_time

def train_transformer(model, train_loader, val_loader, epochs, lr, device):
    print("\n--- Starting Transformer Training ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_path = 'results/temp_best_transformer.pt'
    os.makedirs('results', exist_ok=True)

    patience = 5
    static_epochs = 0

    start_time = time.perf_counter()  # Start Training Timer

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Transformer Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = model(x)
                loss = criterion(output, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    output = model(x)
                    val_loss += criterion(output, y).item()
        val_loss /= len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            static_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | * Best model saved")
        else:
            static_epochs += 1
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | No improvement for {static_epochs} epochs")
            if static_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break

        scheduler.step(val_loss)

    end_time = time.perf_counter()  # End Training Timer
    total_train_time = end_time - start_time

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model, total_train_time

# Evaluation & Benchmarking
def evaluate_and_benchmark(model, test_loader, model_name="Model"):
    model.eval()
    preds, actuals = [], []
    
    print(f"\nWarming up {model_name}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= 5: break
            x = x.to(device)
            _ = model(x)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"{model_name} Inference"):
            x, y = x.to(device), y.to(device)
            output = model(x).view(y.size())
            preds.append(output.cpu().detach().numpy())
            actuals.append(y.cpu().detach().numpy())
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_inference_time = total_time / len(test_loader)

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    
    test_mse = float(np.mean((preds - actuals) ** 2))
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(np.mean(np.abs(preds - actuals)))
    pred_norms = np.linalg.norm(preds.reshape(-1, 3), axis=1)
    mean_deviation = float(np.mean(np.abs(pred_norms - 1.0)))

    preds_flat = preds.reshape(-1, 3)
    actuals_flat = actuals.reshape(-1, 3)
    norms = np.linalg.norm(preds_flat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    preds_normed = (preds_flat / norms).reshape(preds.shape)

    infidelity_fn = Infidelity()
    preds_normed_t = torch.from_numpy(preds_normed.reshape(-1, 3)).float()
    actuals_t = torch.from_numpy(actuals.reshape(-1, 3)).float()
    mean_infidelity = infidelity_fn(preds_normed_t, actuals_t).item()
    mean_fidelity = 1.0 - mean_infidelity

    print(f"\n--- {model_name} Results ---")
    print(f"Total Inference Time: {total_time:.4f} s")
    print(f"Avg Time per Batch: {avg_inference_time*1000:.4f} ms")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test Fidelity: {mean_fidelity:.6f}")
    
    return {
        'total_time': total_time,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'mean_fidelity': mean_fidelity
    }

if __name__ == '__main__':
    set_seed(42)
    
    # Shared Hyperparameters 
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    INPUT_DIM = 3
    D_MODEL = 32
    N_LAYERS = 3      
    WINDOW_SIZE = 256   
    BATCH_SIZE = 128
    
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1
    # ==========================================
    
    path = "data/synthetic/400k_samples_txp_1551.5_pax_1556.5_polcon_and_fiber_2_1Hz.mat"
    
    if not os.path.exists(path):
        print(f"File not found at {path}. Please adjust path.")
        exit(1)
        
    print(f"Loading data from {path}")
    mat_data = scipy.io.loadmat(path)

    s1_pax = mat_data['s1_pax'].flatten()
    s2_pax = mat_data['s2_pax'].flatten()
    s3_pax = mat_data['s3_pax'].flatten()
    s1_txp = mat_data['s1_txp'].flatten()
    s2_txp = mat_data['s2_txp'].flatten()
    s3_txp = mat_data['s3_txp'].flatten()

    features = np.column_stack([s1_txp, s2_txp, s3_txp])
    targets = np.column_stack([s1_pax, s2_pax, s3_pax])

    # Subset for speed
    MAX_SAMPLES = 100000
    features = features[:MAX_SAMPLES]
    targets = targets[:MAX_SAMPLES]

    # Data Splits (70% Train, 10% Val, 20% Test)
    train_end = int(0.7 * len(features))
    val_end = int(0.8 * len(features))

    train_features = features[:train_end]
    train_targets = targets[:train_end]
    val_features = features[train_end:val_end]
    val_targets = targets[train_end:val_end]
    test_features = features[val_end:]
    test_targets = targets[val_end:]

    # DataLoaders
    train_set = SParameterDataset(train_features, train_targets, WINDOW_SIZE)
    val_set = SParameterDataset(val_features, val_targets, WINDOW_SIZE)
    test_set = SParameterDataset(test_features, test_targets, WINDOW_SIZE)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Instantiate Models
    mamba_model = PolarizationMamba(
        input_dim=INPUT_DIM, 
        d_model=D_MODEL, 
        n_layers=N_LAYERS, 
        system=system_os
    ).to(device)
    
    transformer_model = SParameterPredictionModel(
        input_dim=INPUT_DIM, 
        d_model=D_MODEL, 
        n_heads=TRANSFORMER_HEADS, 
        n_layers=N_LAYERS,
        dropout=TRANSFORMER_DROPOUT,
        max_seq_len=WINDOW_SIZE
    ).to(device)

    print(f"\nMamba Parameters: {sum(p.numel() for p in mamba_model.parameters())}")
    print(f"Transformer Parameters: {sum(p.numel() for p in transformer_model.parameters())}")

    # --- 1. TRAIN BOTH MODELS ---
    mamba_model, mamba_train_time = train_mamba(mamba_model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)
    transformer_model, transformer_train_time = train_transformer(transformer_model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)

    # --- 2. BENCHMARK & EVALUATE ---
    mamba_metrics = evaluate_and_benchmark(mamba_model, test_loader, "Mamba")
    transformer_metrics = evaluate_and_benchmark(transformer_model, test_loader, "Transformer")

    # --- 3. FINAL COMPARISON ---
    print("\nFinal Comparison Summary:")
    print(f"{'Metric':<20} | {'Mamba':<12} | {'Transformer':<12}")
    print("-" * 48)
    print(f"{'Test MSE':<20} | {mamba_metrics['test_mse']:<12.6f} | {transformer_metrics['test_mse']:<12.6f}")
    print(f"{'Test RMSE':<20} | {mamba_metrics['test_rmse']:<12.6f} | {transformer_metrics['test_rmse']:<12.6f}")
    print(f"{'Test Fidelity':<20} | {mamba_metrics['mean_fidelity']:<12.6f} | {transformer_metrics['mean_fidelity']:<12.6f}")
    print(f"{'Training Time (s)':<20} | {mamba_train_time:<12.4f} | {transformer_train_time:<12.4f}")
    print(f"{'Inference Time (s)':<20} | {mamba_metrics['total_time']:<12.4f} | {transformer_metrics['total_time']:<12.4f}")
    
    # Evaluate Training Winner
    if mamba_train_time < transformer_train_time:
        t_speedup = transformer_train_time / mamba_train_time
        print(f"\nMamba is {t_speedup:.2f}x faster.")
    else:
        t_speedup = mamba_train_time / transformer_train_time
        print(f"\nTransformer is {t_speedup:.2f}x faster.")

    # Evaluate Inference Winner
    if mamba_metrics['total_time'] < transformer_metrics['total_time']:
        i_speedup = transformer_metrics['total_time'] / mamba_metrics['total_time']
        print(f"Mamba is {i_speedup:.2f}x faster.")
    else:
        i_speedup = mamba_metrics['total_time'] / transformer_metrics['total_time']
        print(f"Transformer is {i_speedup:.2f}x faster.")