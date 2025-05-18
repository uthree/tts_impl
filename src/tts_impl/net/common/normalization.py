from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.base.state import PointwiseModule, StatefulModule


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


class DynamicTanh(nn.Module, PointwiseModule):
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


def _sequential_ema(
    x: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor
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
    y = x * alpha + h * (1 - alpha)
    h_out = y
    return y, h_out


def _parallel_ema(
    x: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor
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

    batch_size, seq_len, d_model = x.Shape
    device = x.device
    dtype = x.dtype

    arange_t = torch.arange(seq_len, device=device, dtype=dtype).view(1, -1, 1)

    alpha_star_t = alpha**arange_t
    beta_star_t = alpha**-arange_t

    x_star = torch.cumsum(x * beta_star_t, dim=1)

    term1 = (1.0 - alpha) * alpha_star * x_star
    term2 = alpha_star * alpha * h

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
                self.register_buffer("alpha_buffer", torch.ones(1, 1, 1) * alpha)
        self.d_model = d_model
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
        if self.alpha_trainable:
            alpha = torch.sigmoid(self.alpha_logit)
        else:
            alpha = self.alpha_buffer

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
        dtype = x.dtype

        h = h.float()
        x = x.float()
        batch_size, seq_len, _ = x.shape

        h_mu_initial, h_sigma_initial = torch.chunk(h, 2, dim=2)  # (batch, 1, 1) each

        if self.alpha_trainable:
            current_alpha = torch.sigmoid(self.alpha_logit)  # (1,1,1)
        else:
            current_alpha = self.alpha_buffer  # (1,1,1)

        # Calculate mean and stddev for each point in the sequence
        mu = x.mean(dim=2, keepdim=True)  # (batch, seq_len, 1)
        sigma = x.std(dim=2, keepdim=True)  # (batch, seq_len, 1)

        # Using torch.isclose for robust floating point comparison
        # Ensure current_alpha is on the same device and dtype for comparison tensor
        zero_tensor = torch.tensor(
            0.0, device=current_alpha.device, dtype=current_alpha.dtype
        )
        one_tensor = torch.tensor(
            1.0, device=current_alpha.device, dtype=current_alpha.dtype
        )

        if torch.isclose(
            current_alpha.squeeze(), zero_tensor, atol=self.eps
        ):  # alpha = 0 case
            ema_mu_seq = mu
            ema_sigma_seq = sigma
        elif torch.isclose(
            current_alpha.squeeze(), one_tensor, atol=self.eps
        ):  # alpha = 1 case
            ema_mu_seq = h_mu_initial.expand(-1, seq_len, -1)
            ema_sigma_seq = h_sigma_initial.expand(-1, seq_len, -1)
        else:  # 0 < alpha < 1 case
            arange_t = torch.arange(seq_len, device=x.device, dtype=x.dtype).view(
                1, -1, 1
            )  # (1, seq_len, 1)

            # Calculate powers of alpha: alpha^t = [alpha^0, alpha^1, ..., alpha^{L-1}]
            # This broadcasts current_alpha (1,1,1) with arange_t (1,seq_len,1)
            powers_of_alpha_t = current_alpha**arange_t

            # Calculate inverse powers of alpha: alpha^(-t) = [alpha^0, alpha^-1, ..., alpha^{-(L-1)}]
            # WARNING: Numerical stability issue here!
            # If alpha is small and t is large, alpha^(-t) can become extremely large and overflow.
            # For example, (1e-5)^(-100) = 1e500, which overflows float64.
            # This method is numerically stable only for limited seq_len or alpha close to 1.
            inv_powers_of_alpha_t = current_alpha**-arange_t

            mu_scaled = mu * inv_powers_of_alpha_t  # (B,L,1) * (1,L,1) -> (B,L,1)
            sigma_scaled = sigma * inv_powers_of_alpha_t

            cumsum_mu_scaled = torch.cumsum(mu_scaled, dim=1)
            cumsum_sigma_scaled = torch.cumsum(sigma_scaled, dim=1)
            # cumsum_mu_scaled contains \sum_{k=0}^{t} (mu_k / alpha^k) at each position t

            # Term 1: (1-alpha) * alpha^t * cumsum(x_k / alpha^k)
            term1_mu = (1.0 - current_alpha) * powers_of_alpha_t * cumsum_mu_scaled
            term1_sigma = (
                (1.0 - current_alpha) * powers_of_alpha_t * cumsum_sigma_scaled
            )

            # Term 2: alpha^(t+1) * y_{-1}
            # powers_of_alpha_t_plus_1 = alpha^(t+1) = [alpha^1, alpha^2, ..., alpha^L]
            powers_of_alpha_t_plus_1 = (
                powers_of_alpha_t * current_alpha
            )  # More stable than current_alpha ** (arange_t + 1) if arange_t is large

            term2_mu = (
                powers_of_alpha_t_plus_1 * h_mu_initial
            )  # (1,L,1) * (B,1,1) -> (B,L,1)
            term2_sigma = powers_of_alpha_t_plus_1 * h_sigma_initial

            ema_mu_seq = term1_mu + term2_mu
            ema_sigma_seq = term1_sigma + term2_sigma

        # Normalize x
        # x: (batch, seq_len, d_model)
        # ema_mu_seq, ema_sigma_seq: (batch, seq_len, 1)
        x_norm = (x - ema_mu_seq) / torch.clamp_min(ema_sigma_seq, min=self.eps)

        # Scale, shift
        if self.elementwise_affine:
            x_norm = x_norm * self.gamma + self.beta  # gamma/beta are (1,1,d_model)

        # Determine the final state h_final for the next segment/iteration
        if seq_len > 0:
            h_final_mu = ema_mu_seq[:, -1:, :]  # (batch, 1, 1)
            h_final_sigma = ema_sigma_seq[:, -1:, :]  # (batch, 1, 1)
        else:  # Should have been caught by seq_len == 0 check earlier, but for safety:
            h_final_mu = h_mu_initial
            h_final_sigma = h_sigma_initial

        h_final = torch.cat([h_final_mu, h_final_sigma], dim=2)  # (batch, 1, 2)

        return x_norm.to(dtype), h_final.to(dtype)

    def _initial_state(self, x):
        batch_size = x.shape[0]
        h_mu = torch.zeros(batch_size, 1, 1, device=x.device)
        h_sigma = torch.ones(batch_size, 1, 1, device=x.device)
        h = torch.cat([h_mu, h_sigma], dim=2)
        return h
