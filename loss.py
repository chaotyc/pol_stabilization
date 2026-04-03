import torch
import torch.nn as nn
import torch.nn.functional as F

class AngularLoss(nn.Module):
    """
    Computation of loss based on angle difference between predicted and target polarization states.
    Measures the geodesic distance on the unit sphere between two vectors.
    """
    def __init__(self, lambda_reg=0.02):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, input, target):
        # Project everything to the unit sphere
        input_len = torch.norm(input, p=2, dim=-1)
        input_norm = F.normalize(input, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)

        # Compute Cosine Similarity
        cos_sim = (input_norm * target_norm).sum(dim=-1)
        
        # Calculate Cosine Distance (1 - cos_sim)
        # Perfectly aligned = 0 loss. Opposite = 2 loss.
        cos_loss = 1.0 - cos_sim
        
        norm_loss = ((input_len - 1.0) ** 2)
        
        return cos_loss.mean() + self.lambda_reg * norm_loss.mean()
    
class PoincareRegularizedMSE(nn.Module):
    def __init__(self, lambda_reg=0.2):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        # Standard MSE loss
        mse = self.mse_loss(input, target)

        # L2 norm regularization to penalize deviations from unit sphere
        input_norm = torch.norm(input, p=2, dim=1)
        reg = ((input_norm - 1) ** 2).mean()
        
        return mse + self.lambda_reg * reg

class Infidelity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        """
        input: (batch, 3) - The model's predicted Stokes vector (must be L2 normalized)
        target: (batch, 3) - The target quantum Stokes vector (must be L2 normalized)
        """
        dot_product = torch.sum(input * target, dim=-1)
        fidelity = 0.5 * (1.0 + dot_product)
        infidelity = 1.0 - fidelity
        return torch.mean(infidelity)