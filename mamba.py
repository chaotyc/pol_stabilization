import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Initialize A parameter
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True # Keep A_log out of optimizer weight decay

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True # Keep D out of optimizer weight decay

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

        # SSM Parameters
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
    def __init__(self, input_dim: int, d_model: int, n_layers: int, system: str):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            # If on Linux and has mamba-ssm package, use it
            # Otherwise, fall back to the local PyTorch version
            if system == "Linux":
                print("Initializing Mamba Block (CUDA Optimized)")
                from mamba_ssm import Mamba
                self.layers.append(
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                )
            else:
                print("Initializing Mamba Block (PyTorch Implementation)")
                
                # Windows or fallback
                self.layers.append(
                    MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
                )

        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x) + x
        x = self.norm_f(x)
        out = self.head(x[:, -1, :])
        return out.view(-1, 1, 3)
