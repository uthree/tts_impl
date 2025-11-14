import torch
from torch import nn as nn
from torch.nn import functional as F
from tts_impl.net.base.stateful import StatefulModule
from tts_impl.utils.config import derive_config


@torch.jit.script
def parallel_scan_log(log_coeffs, log_values):
    """
    Heinsen parallel scan

    Args:
        log_coeffs: shape=(batch_size, seq_len, d_model)
        log_values: shape=(batch_size, seq_len + 1, d_model)

    Returns:
        log_h
    """
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)


@torch.jit.script
def g(x: torch.Tensor) -> torch.Tensor:
    """
    Activation function for minGRU.
    """
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


@torch.jit.script
def log_g(x: torch.Tensor) -> torch.Tensor:
    """
    Activation function for minGRU's log-space parallel scan.
    """
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


@torch.jit.script
def mingru_parallel(
    z: torch.Tensor, h: torch.Tensor, h_prev: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel forward pass for minGRU.

    Args:
        z: gate signal, shape=(batch_size, seq_len, d_hidden)
        h: hidden state each time, shape=(batch_size, seq_len, d_hidden)
        h_prev: initial hidden state, shape=(batch_size, 1, d_hidden)

    Retruns:
        y: output signal, shape=(batch_size, seq_len, d_hidden)
        h_next: las hidden state, shape=(batch_size, 1, d_hidden)
    """
    log_z = -F.softplus(-z)
    log_coeffs = -F.softplus(z)
    log_tilde_h = log_g(h)
    log_h_prev = torch.log(h_prev)
    h = parallel_scan_log(
        log_coeffs, torch.cat([log_h_prev, log_z + log_tilde_h], dim=1)
    )
    y = h[:, 1:]
    h_next = y[:, -1:]
    return y, h_next


@torch.jit.script
def mingru_sequential(
    z: torch.Tensor, h: torch.Tensor, h_prev: torch.Tensor
) -> torch.Tensor:
    """
    Sequential forward pass for minGRU.

    Args:
        z: gate signal, shape=(batch_size, 1, d_hidden)
        h: input signal shape=(batch_size, 1, d_hidden)
        h_prev: previous hidden state, shape=(batch_size, 1, d_hidden)

    Returns:
        h_next: output / next hidden state, shape=(batch_size, 1, d_hidden)
    """
    z = torch.sigmoid(z)
    h_tilde = g(h)
    h_next = (1 - z) * h_prev + z * h_tilde
    return h_next


@derive_config
class MinGRU(StatefulModule):
    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        bias: bool = True,
        d_cond: int | None = None
    ):
        """
        Unofficial implementation of [minGRU](https://arxiv.org/abs/2410.01201v1).

        Args:
            d_model: int, input dimension
            d_hidden: int | None, hidden dimension, if `None` given, set same to `d_model` automatically.
            bias: bool, if True is given, add bias to Linear module., default is `True`
        """
        super().__init__()
        self.d_model = d_model
        if d_hidden is None:
            d_hidden = d_model
        self.d_hidden = d_hidden
        self.linear_h = nn.Linear(d_model, d_hidden, bias=bias)
        self.linear_z = nn.Linear(d_model, d_hidden, bias=bias)
        self.d_cond = d_cond
        if d_cond is not None:
            self.cond = nn.Linear(d_cond, d_model * 2)

    def _sequential_forward(
        self, x: torch.Tensor, h_prev: torch.Tensor, cond: torch.Tensor| None=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cond is not None and self.d_cond is not None:
            mu, sigma = self.cond(cond).chunk(2, dim=2)
            x = x * torch.exp(sigma) + mu
        z = self.linear_z(x)
        h = self.linear_h(x)
        h = mingru_sequential(z, h, h_prev)
        return h, h

    def _parallel_forward(
        self, x: torch.Tensor, h_prev: torch.Tensor, cond: torch.Tensor| None=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cond is not None and self.d_cond is not None:
            mu, sigma = self.cond(cond).chunk(2, dim=2)
            x = x * torch.exp(sigma) + mu
        z = self.linear_z(x)
        h = self.linear_h(x)
        y, h_next = mingru_parallel(z, h, h_prev)
        return y, h_next

    def _initial_state(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.shape[0], 1, self.d_hidden, device=x.device, dtype=torch.float
        )
