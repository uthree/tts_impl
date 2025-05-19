from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.base.stateful import StatefulModule


class DepthwiseCachedConv(StatefulModule):
    """
    Depthwise causal convoluition with cache for streaming inference.
    """

    def __init__(self, d_model: int, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        assert kernel_size > 1, "kernel size should greater than 1"
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size, groups=d_model, bias=bias, padding=0
        )

    def _sequential_forward(self, x, h):
        x, h = self._parallel_forward(x, h)
        return x, h

    def _parallel_forward(self, x, h) -> Tuple[torch.Tensor, torch.Tensor]:
        b, l, d = x.shape
        d = self.d_model
        k = self.kernel_size
        x = x.transpose(1, 2)  # [b, l, d] -> [b, d, l]
        h = h.view(b, d, (k - 1))  # [b, 1, (k-1) * d] -> [b, d, (k-1)]
        x = torch.cat([h, x], dim=2)  # [b, d, l+(k-1)]
        h = x[:, :, -(k - 1) :]
        x = self.conv(x)  # [b, d, l+(k-1)] -> [b, d, l]
        h = h.reshape(b, 1, (k - 1) * d)  # [b, d, (k-1)] -> [b, 1, (k-1) * d]
        x = x.transpose(1, 2)  # [b, d, l] -> [b, l, d]
        return x, h

    def _initial_state(self, x):
        device = x.device
        dtype = x.dtype
        b, l, d = x.shape
        d = self.d_model
        k = self.kernel_size
        h = torch.zeros(size=(b, 1, (k - 1) * d), device=device, dtype=dtype)
        return h
