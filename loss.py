import numpy as np
from numpy import pi
import torch.nn as nn
import torch

class WeightedMeanSquaredLoss(nn.Module):
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
        loss_1 = (preds - targets)**2
        weighted_loss_1 = loss_1 @ self.weights
        weighted_mean_loss = torch.mean(weighted_loss_1)
        
        if be_static is not None and len(be_static)>0:
            loss_2 = (be_static)**2
            weighted_loss_2 = loss_2 @ self.weights
            weighted_loss_mean_loss_2 =torch.mean(weighted_loss_2)*0.1
            weighted_mean_loss = weighted_mean_loss + weighted_loss_mean_loss_2
            
        
        return weighted_mean_loss
       
    
