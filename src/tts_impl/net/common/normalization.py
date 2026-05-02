import torch
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.net.base.stateful import StatefulModule


class LayerNorm1d(nn.Module):
    """
    layer normalization for 1d sequence.
    """

    def __init__(
        self, channels: int, eps: float = 1e-12, elementwise_affine: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(1, channels, 1))
            self.gamma = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, channels, length)

        Returns:
            x: shape=(batch_size, channels, length)
        """
        dtype = x.dtype
        x = x.to(torch.float)
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mu) / sigma
        if self.elementwise_affine:
            x = x * self.gamma + self.beta
        x = x.to(dtype)
        return x


class RMSNorm1d(nn.Module):
    """
    RMS normalization for 1d sequence.
    """

    def __init__(
        self, channels: int, eps: float = 1e-12, elementwise_scale: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.elementwise_scale = elementwise_scale
        if elementwise_scale:
            self.gamma = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, channels, length)

        Returns:
            x: shape=(batch_size, channels, length)
        """
        dtype = x.dtype
        x = x.to(torch.float)
        rms = torch.sqrt(self.eps + torch.std(x, dim=1))
        x = x / rms
        if self.elementwise_scale:
            x = x * self.gamma
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, channels: int, eps: float = 1e-12):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, channels, length)

        Returns:
            x: shape=(batch_size, channels, length)
        """
        dtype = x.dtype
        x = x.float()
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=2, keepdim=True) + self.eps)
        x = self.gamma * (x * nx) + self.beta + x
        x = x.to(dtype)
        return x


class LayerNorm(nn.Module):
    """
    layer normalization for sequential model
    """

    def __init__(
        self, d_model: int, eps: float = 1e-12, elementwise_affine: bool = True
    ):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, d_model))
            self.gamma = nn.Parameter(torch.ones(1, 1, d_model))
        self.dim = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, seq_len, d_model)

        Returns:
            x: shape=(batch_size, seq_len, d_model)
        """
        dtype = x.dtype
        x = x.to(torch.float)
        mu = x.mean(dim=(1, 2), keepdim=True)
        sigma = x.std(dim=(1, 2), keepdim=True) + self.eps
        x = (x - mu) / sigma
        if self.elementwise_affine:
            x = x * self.gamma + self.beta
        x = x.to(dtype)
        return x


class DynamicTanh(StatefulModule):
    """
    dynamic tanh layer for sequential model instead of normalization.
    reference: https://arxiv.org/abs/2503.10622
    """

    def __init__(self, d_model: int, alpha: float = 0.5, elementwise_affine=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, 1) * alpha)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, d_model))
            self.gamma = nn.Parameter(torch.ones(1, 1, d_model))

    def _parallel_forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, seq_len, d_model)
            h: dummy state, shape=(batch_size, 1, 0)

        Returns:
            x: shape=(batch_size, seq_len, d_model)
            h: dummy state, shape=(batch_size, 1, 0)
        """
        x = F.tanh(self.alpha * x)
        if self.elementwise_affine:
            x = x * self.gamma + self.beta
        return x, h

    def _initial_state(self, x) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1, 0), device=x.device)


class GlobalResponseNorm(nn.Module):
    """
    global response normalization for sequential model.
    """

    def __init__(self, d_model: int, eps=1e-12):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, 1, d_model))
        self.gamma = nn.Parameter(torch.zeros(1, 1, d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, seq_len, d_model)

        Returns:
            x: shape=(batch_size, seq_len, d_model)
        """
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x
