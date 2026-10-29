import math
import platform
import warnings
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
        dt_max = 0.1
        dt_min = 0.001
        dt_init_floor = 1e-4
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

        h = x.new_zeros(batch_size, self.d_inner, self.d_state)
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

def build_mamba_layers(d_model, n_layers, system=None, device=None, backend="auto"):
    """Use optional CUDA kernels only on a compatible target; CPU always has a fallback."""
    if backend not in {"auto", "pytorch", "cuda"}:
        raise ValueError(f"Unknown Mamba backend: {backend}")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    compatible = (system or platform.system()) == "Linux" and target.type == "cuda" and torch.cuda.is_available()
    block_type = MambaBlock
    selected = "pytorch"
    if backend == "cuda" and not compatible:
        raise ValueError("The cuda Mamba backend requires Linux and a CUDA device; use pytorch or auto")
    if backend != "pytorch" and compatible:
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as exc:
            if backend == "cuda":
                raise RuntimeError("Install compatible mamba-ssm CUDA kernels or use --backend pytorch") from exc
            warnings.warn(f"mamba-ssm unavailable; using PyTorch Mamba: {exc}", RuntimeWarning)
        else:
            block_type, selected = Mamba, "cuda"
    layers = nn.ModuleList([
        block_type(d_model=d_model, d_state=16, d_conv=4, expand=2)
        for _ in range(n_layers)
    ])
    return layers, selected


class PolarizationMamba(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_layers: int, system=None, device=None, backend="auto"):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)

        self.layers, self.backend = build_mamba_layers(d_model, n_layers, system, device, backend)

        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x) + x
        x = self.norm_f(x)
        out = self.head(x[:, -1, :])
        return out.view(-1, 1, 3)

class PolarizationMambaSO3(nn.Module):
    feature_multiplier = 1

    def __init__(self, input_dim: int, d_model: int, n_layers: int, system=None, device=None, backend="auto"):
        super().__init__()
        # input_dim describes the raw coordinates; variants construct extra features internally.
        self.embedding = nn.Linear(input_dim * self.feature_multiplier, d_model)

        self.layers, self.backend = build_mamba_layers(d_model, n_layers, system, device, backend)

        self.norm_f = nn.LayerNorm(d_model)

        # Predict a 3D vector representing the rotation generator Omega
        self.head = nn.Linear(d_model, 3)

    def get_skew_symmetric(self, omega):
        """
        Maps a 3D generator vector to the so(3) Lie algebra (a 3x3 skew-symmetric matrix).
        omega: (batch, 3) -> [wx, wy, wz]
        """
        batch_size = omega.shape[0]
        zero = omega.new_zeros(batch_size)
        wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]

        # Skew-symmetric matrix J
        # [  0, -wz,  wy]
        # [ wz,   0, -wx]
        # [-wy,  wx,   0]
        M = torch.stack([
            zero, -wz, wy,
            wz, zero, -wx,
            -wy, wx, zero
        ], dim=1).view(batch_size, 3, 3)
        return M

    def _prepare_features(self, x):
        return x[:, 1:, :] - x[:, :-1, :]

    def encode_window(self, x):
        # x shape: (batch, window_size, 3) representing absolute TxP control states
        if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] != 3:
            raise ValueError(f"{type(self).__name__} requires shape (batch, window_size >= 2, 3)")

        # 1. Input Transformation: Construct features from the observed TxP window.
        embed = self.embedding(self._prepare_features(x))
        for layer in self.layers:
            embed = layer(embed) + embed
        return self.norm_f(embed)[:, -1, :]

    def rotate_control(self, x, omega):
        # 3. Output Transformation: Map generator to the SO(3) Lie group
        skew_matrix = self.get_skew_symmetric(omega)
        # torch.matrix_exp natively computes the matrix exponential and is fully differentiable
        R_relative = torch.matrix_exp(skew_matrix) # Shape: (batch, 3, 3)

        # 4. Final Prediction Mapping: Rotate the control signal
        # Grab the most recent absolute control signal and ensure it is mathematically normalized
        current_txp = F.normalize(x[:, -1, :], p=2, dim=-1) # Shape: (batch, 3)

        # Apply the predicted rotation matrix to the control signal
        # Batch matrix multiplication: (batch, 3, 3) x (batch, 3, 1)
        predicted_pax = torch.bmm(R_relative, current_txp.unsqueeze(-1)).squeeze(-1)

        return predicted_pax.view(-1, 1, 3)

    def forward(self, x):
        omega = self.head(self.encode_window(x))
        return self.rotate_control(x, omega)


class PolarizationMambaSO3Combined(PolarizationMambaSO3):
    """Predict SO3 rotations from absolute TxP coordinates and their differences.

    Accepts raw (batch, window_size, 3) windows. Each of the window_size - 1
    feature vectors is [s1(t), s2(t), s3(t), ds1(t), ds2(t), ds3(t)], with
    ds(t) = s(t) - s(t-1). The coordinates and differences refer to the same
    observed time step; no future samples are used.
    """

    feature_multiplier = 2

    def _prepare_features(self, x):
        return torch.cat((x[:, 1:, :], super()._prepare_features(x)), dim=-1)


class PolarizationMambaSO3CombinedTime(PolarizationMambaSO3Combined):
    """Condition the six-feature model's rotation head on the target timestamp.

    Time is normalized by a fixed origin/scale fitted to the training prefix.
    It bypasses input differencing, and is never clipped at the training range.
    """

    requires_time = True

    def __init__(self, input_dim: int, d_model: int, n_layers: int, system=None, device=None,
                 backend="auto", time_origin=0.0, time_scale=1.0):
        if not math.isfinite(time_origin) or not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("Time origin must be finite and time_scale must be finite and positive")
        super().__init__(input_dim, d_model, n_layers, system, device, backend)
        previous_head = self.head
        self.head = nn.Linear(d_model + 1, 3)
        # Match the baseline's initial predictions; the extra time coefficient starts at zero.
        with torch.no_grad():
            self.head.weight[:, :d_model].copy_(previous_head.weight)
            self.head.weight[:, d_model:].zero_()
            self.head.bias.copy_(previous_head.bias)
        self.register_buffer("time_origin", torch.tensor(float(time_origin), dtype=torch.float64))
        self.register_buffer("time_scale", torch.tensor(float(time_scale), dtype=torch.float64))

    def forward(self, x, target_time):
        encoded = self.encode_window(x)
        times = torch.as_tensor(target_time, device=x.device, dtype=self.time_origin.dtype)
        if times.ndim == 0:
            times = times.expand(len(x))
        elif times.shape == (len(x), 1):
            times = times[:, 0]
        if times.shape != (len(x),) or not torch.isfinite(times).all():
            raise ValueError("target_time must be finite and scalar or have one timestamp per window")
        normalized = ((times - self.time_origin) / self.time_scale).to(encoded.dtype)
        omega = self.head(torch.cat((encoded, normalized[:, None]), dim=-1))
        return self.rotate_control(x, omega)


SO3_MODEL_TYPES = {
    "PolarizationMambaSO3": PolarizationMambaSO3,
    "PolarizationMambaSO3Combined": PolarizationMambaSO3Combined,
    "PolarizationMambaSO3CombinedTime": PolarizationMambaSO3CombinedTime,
}
