from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.base.stateful import PointwiseModule, StatefulModule


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


class ExponentialMovingAverage(StatefulModule):
    def __init__(
        self,
        d_model: int,
        alpha: float = 0.99,
        trainable: bool = True,
        initial_value: float = 0.0,
    ):
        super().__init__()
        assert alpha >= 0.0, "`alpha` should be greater than 0 or equal to 0."
        assert alpha <= 1.0, "`alpha` should be less than 1.0 or equal to 0."

        self.d_model = d_model
        self.trainable = trainable
        self.initial_value = initial_value
        alpha_logit = torch.logit(torch.full(size=(1, 1, d_model), fill_value=alpha))
        with torch.no_grad():
            if trainable:
                self.alpha_logit = nn.Parameter(alpha_logit)
            else:
                self.register_buffer("alpha_logit", alpha_logit)

    def _initial_state(self, x):
        device = x.device
        batch_size = x.shape[0]
        return torch.ones(batch_size, 1, self.d_model) * self.initial_value

    def _get_alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def _sequential_forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        calculate exponential moving average.

        Args:
            x: input signal, shape=(batch_size, 1, d_model)
            h: previous state, shape=(batch_size, 1, d_model)
            alpha: ema coeff., shape=(batch_size, 1, d_model)
        Returns
            y: output signal, shape=(batch_size, 1, d_model)
            h_out: last state, shape=(batch_size, 1, d_model)
        """
        alpha = self._get_alpha()
        y = h * alpha + x * (1 - alpha)
        h_out = y
        return y, h_out

    def _parallel_forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        calculate exponential moving average for each time using cumsum for performance.

        Args:
            x: input signal, shape=(batch_size, seq_len, d_model)
            h: previous state, shape=(batch_size, 1, d_model)
            alpha: ema coeff., shape=(batch_size, 1, d_model)
        Returns
            y: output signal, shape=(batch_size, seq_len, d_model)
            h_out: last state, shape=(batch_size, 1, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        dtype = x.dtype
        alpha = self._get_alpha()

        arange_t = torch.arange(seq_len, device=device, dtype=dtype).view(1, -1, 1)

        alpha_star_t = alpha**arange_t
        beta_star_t = alpha**-arange_t

        x_star = torch.cumsum(x * beta_star_t, dim=1)

        term1 = (1.0 - alpha) * alpha_star_t * x_star
        term2 = alpha_star_t * alpha * h

        y = term1 + term2
        h_out = y[:, -1:, :]
        return y, h_out


class EmaLayerNorm(StatefulModule):
    """
    layer normalization for sequential model with streaming inference.
    apply exponential moving average for mean and standard deviation.
    """

    def __init__(
        self,
        d_model: int,
        elementwise_affine: bool = True,
        alpha_mu: float = 0.99,
        alpha_sigma: float = 0.99,
        alpha_trainable: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.d_model = d_model
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, d_model))
            self.gamma = nn.Parameter(torch.ones(1, 1, d_model))
        self.ema_mu = ExponentialMovingAverage(
            d_model=1, alpha=alpha_mu, trainable=alpha_trainable, initial_value=0.0
        )
        self.ema_sigma = ExponentialMovingAverage(
            d_model=1, alpha=alpha_sigma, trainable=alpha_trainable, initial_value=1.0
        )
        self.eps = eps

    def _both_forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape=(batch_size, seq_len, d_model)
            h: shape=(batch_size, 1, 2)

        Returns:
            x: shape=(batch_size, seq_len, d_model)
            h: shape(batch_size, 1, 2)
        """
        # expand state to mean and standard deviation.
        h_mu, h_sigma = torch.chunk(h, 2, dim=2)

        # calculate mean and standard deviation of x
        mu = x.mean(dim=2, keepdim=True)
        sigma = x.std(dim=2, keepdim=True)

        h_mu, h_mu_last = self.ema_mu(mu, h_mu)
        h_sigma, h_sigma_last = self.ema_sigma(sigma, h_sigma)

        # normalize
        x = (x - h_mu) / torch.clamp_min(h_sigma, min=self.eps)

        # scale, shift
        if self.elementwise_affine:
            x = x * self.gamma + self.beta

        # pack h_mu, h_sigma to h
        h = torch.cat([h_mu_last, h_sigma_last], dim=2)
        return x, h

    def _parallel_forward(self, x, h):
        return self._both_forward(x, h)

    def _sequential_forward(self, x, h):
        return self._both_forward(x, h)

    def _initial_state(self, x):
        batch_size = x.shape[0]
        h_mu = torch.mean(x[:, :1], dim=2, keepdim=True)
        h_sigma = torch.std(x[:, :1], dim=2, keepdim=True)
        h = torch.cat([h_mu, h_sigma], dim=2)
        return h


class EmaInstanceNorm(StatefulModule):
    """
    instance normalization for sequential model with streaming inference.
    apply exponential moving average for mean and standard deviation.
    """

    def __init__(
        self,
        d_model: int,
        elementwise_affine: bool = True,
        alpha_mu: float = 0.99,
        alpha_sigma: float = 0.99,
        alpha_trainable: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.d_model = d_model
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, d_model))
            self.gamma = nn.Parameter(torch.ones(1, 1, d_model))
        self.ema_mu = ExponentialMovingAverage(
            d_model=d_model,
            alpha=alpha_mu,
            trainable=alpha_trainable,
            initial_value=0.0,
        )
        self.ema_sigma = ExponentialMovingAverage(
            d_model=1,
            alpha=alpha_sigma,
            trainable=alpha_trainable,
            initial_value=1.0,
        )
        self.eps = eps

    def _both_forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape=(batch_size, seq_len, d_model)
            h: shape=(batch_size, 1, d_model + 1)

        Returns:
            x: shape=(batch_size, seq_len, d_model)
            h: shape(batch_size, 1, d_model + 1)
        """
        h_mu, h_sigma = torch.split(h, [self.d_model, 1], dim=2)

        # normalize and update EMA.
        h_mu, h_mu_last = self.ema_mu(x, h_mu)
        x = x - h_mu
        sigma = torch.clamp_min(x.pow(2).mean(dim=2, keepdim=True), 1).sqrt()
        h_sigma, h_sigma_last = self.ema_sigma(sigma, h_sigma)
        x = x / torch.clamp_min(h_sigma, min=1.0)

        # scale, shift
        if self.elementwise_affine:
            x = x * self.gamma + self.beta

        # pack h_mu, h_sigma to h
        h = torch.cat([h_mu_last, h_sigma_last], dim=2)
        return x, h

    def _parallel_forward(self, x, h):
        return self._both_forward(x, h)

    def _sequential_forward(self, x, h):
        return self._both_forward(x, h)

    def _initial_state(self, x):
        h_mu = x[:, :1]
        h_sigma = torch.clamp_min(
            x[:, :1].pow(2).mean(dim=2, keepdim=True).sqrt(), min=1.0
        )
        h = torch.cat([h_mu, h_sigma], dim=2)
        return h
