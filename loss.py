import numpy as np
from numpy import pi
import torch.nn as nn
import torch

class WeightedMeanSquaredLoss(nn.Module):
    """Weighted MSE loss with optional static obstacle penalty.

    Applies a time-weighted mean squared error, where later timesteps can be
    weighted higher (cubic weighting). Optionally penalizes non-zero predictions
    for static obstacles.

    Args:
        horizon (int): Prediction horizon length. Default: 20.
        device (str): Device for tensors. Default: 'cpu'.
        no_weight (bool): If True, use uniform weights. Default: False.

    Inputs:
        preds (Tensor): Predicted controls (N, horizon*2).
        targets (Tensor): Ground truth controls (N, horizon*2).
        be_static (Tensor, optional): Static obstacle predictions (N, horizon*2) for regularization.

    Returns:
        Tensor: Scalar loss value.
    """
    def __init__(self, horizon = 20, device = 'cpu', no_weight=False):
        super().__init__()
        
        self.device = device
        
        if no_weight:
            self.weights = np.ones(horizon)
        elif horizon <= 10: 
            self.weights = np.flip(np.array(range(horizon))**3)
        else:
            self.weights = np.flip(np.array(range(10))**3)
            self.weights = np.append(self.weights, [1]*(horizon-10))
  
        self.weights = torch.from_numpy(self.weights / sum(self.weights)).type(torch.float32)
        self.weights = self.weights.repeat_interleave(2).to(self.device)
    
    def forward(self, preds, targets, be_static=None):
        """Compute weighted MSE, plus optional static obstacle regularization.

        Args:
            preds (Tensor): Predictions (N, horizon*2).
            targets (Tensor): Ground truth (N, horizon*2).
            be_static (Tensor, optional): Static obstacle predictions.

        Returns:
            Tensor: Scalar loss.
        """
        loss_1 = (preds - targets)**2
        weighted_loss_1 = loss_1 @ self.weights
        weighted_mean_loss = torch.mean(weighted_loss_1)
        
        if be_static is not None and len(be_static)>0:
            loss_2 = (be_static)**2
            weighted_loss_2 = loss_2 @ self.weights
            weighted_loss_mean_loss_2 =torch.mean(weighted_loss_2)*0.1
            weighted_mean_loss = weighted_mean_loss + weighted_loss_mean_loss_2
            
        
        return weighted_mean_loss
       
