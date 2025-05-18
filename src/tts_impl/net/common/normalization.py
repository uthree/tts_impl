import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.base.state import Stateful, Pointwise
from typing import Optional, Tuple


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
        mu = x.mean(dim=(1, 2), keepdim=True)
        sigma = x.std(dim=(1, 2), keepdim=True) + self.eps
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
        rms = torch.sqrt(self.eps + torch.std(x, dim=(1, 2)))
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
        self.dim = dim
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


class DynamicTanh(nn.Module, Pointwise):
    """
    dynamic tanh layer for sequential model instead of normalization.
    reference: https://arxiv.org/abs/2503.10622
    """

    def __init__(self, d_model: int, alpha: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, 1) * alpha)
        self.beta = nn.Parameter(torch.zeros(1, 1, d_model))
        self.gamma = nn.Parameter(torch.ones(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, seq_len, d_model)

        Returns:
            x: shape=(batch_size, seq_len, d_model)
        """
        return F.tanh(self.alpha * x) * self.gamma + self.beta


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


class EmaLayerNorm(nn.Module, Stateful):
    """
    layer normalization for sequential model with streaming inference.
    apply exponential moving average for mean and standard deviation.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-12,
        elementwise_affine: bool = True,
        alpha: float = 0.99,
        alpha_trainable: bool = True,
    ):
        super().__init__()
        assert alpha >= 0.0, "`alpha` should be greater than 0 or equal to 0."
        assert alpha <= 1.0, "`alpha` should be less than 1.0 or equal to 0."

        with torch.no_grad():
            self.elementwise_affine = elementwise_affine
            self.alpha_trainable = alpha_trainable
            if elementwise_affine:
                self.beta = nn.Parameter(torch.zeros(1, 1, d_model))
                self.gamma = nn.Parameter(torch.ones(1, 1, d_model))

            if alpha_trainable:
                self.alpha_logit = nn.Parameter(
                    torch.logit(torch.ones(1, 1, 1) * alpha)
                )
            else:
                self.register_buffer("alpha", torch.ones(1, 1, 1) * alpha)
        self.dim = dim
        self.eps = eps

    def _sequential_forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape=(batch_size, 1, d_model)
            h: shape=(batch_size, 1, 2)

        Returns:
            x: shape=(batch_size, 1, d_model)
            h: shape(batch_size, 1, 2)
        """
        # expand state to mean and standard deviation.
        h_mu, h_sigma = torch.chunk(h, 2, dim=2)

        # calculate alpha
        if alpha_trainable:
            alpha = torch.sigmoid(self.alpha_logit)
        else:
            alpha = self.alpha

        # calculate mean and standard deviation of x
        mu = x.mean(dim=2, keepdim=True)
        sigma = x.std(dim=2, keepdim=True)

        # update state
        h_mu = h_mu * alpha + mu * (1 - alpha)
        h_sigma = h_sigma * alpha + sigma * (1 - alpha)

        # normalize
        x = (x - h_mu) / torch.clamp_min(h_sigma, min=self.eps)

        # scale, shift
        if self.elementwise_affine:
            x = x * self.gamma + self.beta

        # pack h_mu, h_sigma to h
        h = torch.cat([h_mu, h_sigma], dim=2)
        return x, h

    def _parallel_forward(self, x, h):
        batch_size, seq_len, d_model = x.shape

        # expand state to mean and standard deviation.
        h_mu, h_sigma = torch.chunk(h, 2, dim=2)

        # calculate alpha
        if alpha_trainable:
            alpha = torch.sigmoid(self.alpha_logit)
        else:
            alpha = self.alpha

        # calculate mean and stddev for each point
        mu = x.mean(dim=2, keepdim=True)
        sigma = x.std(dim=2, keepdim=True)

        # parallel scan
        t = torch.arange(seq_len, device=x.device)[None, None, :]
        alpha_t = (1 - alpha) * torch.pow(alpha, t)
        mu_cumlative = h_mu * alpha + torch.cumsum(alpha_t * mu, dim=2)
        sigma_cumlative = h_sigma * alpha + torch.cumsum(alpha_t * sigma, dim=2)
        sigma_cumlative = torch.clamp_min(sigma_cumlative, min=self.eps)  # to avoid NaN

        # normalize
        x = (x - mu_cumlative) / sigma_cumlative

        # scale, shift
        x = x * self.gamma + self.beta

        # pack mu, sigma
        h_mu = mu_cumlative[:, -1:, :]
        h_sigma = sigma_cumlative[:, -1:, :]
        h = torch.cat([h_mu, h_sigma], dim=2)
        return x, h

    def _initial_state(self, x):
        batch_size = x.shape[0]
        h_mu = torch.zeros(batch_size, 1, 1, device=x.device)
        h_sigma = torch.ones(batch_size, 1, 1, device=x.device)
        h = torch.cat([h_mu, h_sigma], dim=2)
        return h


class EmaInstanceNorm(nn.Module, Stateful):
    def __init__(self):
        super().__init__()
