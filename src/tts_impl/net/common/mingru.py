from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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


@derive_config
class MinGRU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        p_dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Unofficial implementation of [minGRU](https://arxiv.org/abs/2410.01201v1).

        Args:
            d_model: int, input dimension
            d_hidden: Optional[int], hidden dimension, if `None` given, set same to `d_model` automatically.
            p_dropout: float, probabilities of dropout
            bias: bool, if True is given, add bias to Linear module., default is `True`
        """
        super().__init__()
        self.d_model = d_model
        if d_hidden is None:
            d_hidden = d_model
        self.d_hidden = d_hidden
        self.p_dropout = p_dropout
        self.linear_h = nn.Linear(d_model, d_hidden, bias=bias)
        self.linear_z = nn.Linear(d_model, d_hidden, bias=bias)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        """
        Args:
            x: Tensor, shape=(batch_size, seq_len, d_model), input sequence
            h_prev: Optional[Tensor], shape=(batch_size, 1, d_model), initial state.

        Retrun:
            h: Tensor, shape=(batch_size, seq_len, d_hidden)
        """
        if h_prev is None:
            h_prev = torch.zeros(size=(x.shape[0], 1, self.d_hidden), device=x.device)
        x = F.dropout(x, self.p_dropout)
        if x.shape[1] == 1:
            return self._sequential_forward(x, h_prev)
        else:
            return self._parallel_forward(x, h_prev)

    def _sequential_forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor]):
        z = torch.sigmoid(self.linear_z(x))
        h_tilde = g(self.linear_h(x))
        h = (1 - z) * h_prev + z * h_tilde
        return h

    def _parallel_forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor]):
        k = self.linear_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_tilde_h = log_g(self.linear_h(x))
        log_h_0 = torch.log(h_0)
        h = parallel_scan_log(
            log_coeffs, torch.cat([log_h_0, log_z + log_tilde_h], dim=1)
        )
        print(h)
        h = h[:, 1:]
        return h
