import torch.nn as nn
import torch
import torch.nn.functional as F

class LayerNorm1d(nn.Module):
    """
    layer normalization for 1d sequence.
    """

    def __init__(self, channels, eps=1e-12):
        super().__init__()
        self.channels = channels
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: shape=(batch_size, channels, length)

        Returns:
            x: shape=(batch_size, channels, length)
        """
        dtype = x.dtype
        x = x.to(torch.float)
        mu = x.mean(dim=(1, 2), keepdim=True)
        sigma = x.std(dim=(1, 2), keepdim=True) + self.eps
        x = (x - mu) / sigma
        x = x.to(dtype)
        return x

 
class DynamicTanh1d(nn.Module):
    """
    dynamic tanh layer for 1d-sequence instead of normalization.
    reference: https://arxiv.org/abs/2503.10622
    """

    def __init__(self, channels, alpha: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, 1) * alpha)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        """
        Args:
            x: shape=(batch_size, channels, length)

        Returns:
            x: shape=(batch_size, channels, length)
        """
        return F.tanh(self.alpha * x) * self.gamma + self.beta
    

class GlobalResponseNorm1d(nn.Module):
    """
    global response normalization
    """
    def __init__(self, channels, eps=1e-12):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps


    def forward(self, x):
        """
        Args:
            x: shape=(batch_size, channels, length)

        Returns:
            x: shape=(batch_size, channels, length)
        """
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x