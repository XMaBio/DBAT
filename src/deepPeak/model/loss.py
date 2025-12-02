import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareMSELoss(nn.Module):
    def __init__(self, boundary_weight=5.0):
        super().__init__()
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        with torch.no_grad():
            grad = torch.abs(target[:, 1:] - target[:, :-1])
            weights = torch.cat([grad, torch.zeros_like(grad[:, :1])], dim=1)
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            weights = 1.0 + self.boundary_weight * weights

        loss = weights * (pred - target) ** 2
        return loss.mean()



class RatioLoss(nn.Module):
    """
    Loss function for ratio prediction
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_ratio, true_ratio):
        # Main MSE loss for ratio prediction
        mse_loss = self.mse_loss(pred_ratio, true_ratio)

        # Additional penalty for predicting zeros
        # (encourages learning real patterns)
        zero_penalty = torch.mean(torch.abs(pred_ratio)) * self.alpha

        total_loss = mse_loss + zero_penalty

        return total_loss, mse_loss, zero_penalty
