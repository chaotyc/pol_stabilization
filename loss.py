import torch
import torch.nn as nn
import torch.nn.functional as F

class AngularLoss(nn.Module):
    """
    Computation of loss based on angle difference between predicted and target polarization states.
    Measures the geodesic distance on the unit sphere between two vectors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Project everything to the unit sphere
        input_norm = F.normalize(input, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # Compute Cosine Similarity of u and v (dot product)
        cos_sim = (input_norm * target_norm).sum(dim=1)
        
        # Compute angle theta = arccos(<u, v>)
        theta = torch.acos(cos_sim)
        
        return theta.mean()
    
class PoincareRegularizedMSE(nn.Module):
    def __init__(self, lambda_reg=0.1):
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