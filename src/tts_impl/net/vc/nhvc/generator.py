from typing import Tuple

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from tts_impl.net.base.stateful import StatefulModule, StatefulModuleSequential
from tts_impl.net.common.mingru import MinGRU
from tts_impl.utils.config import derive_config


class NhvcLayer(StatefulModule):
    def __init__(self, d_model: int, gin_channels: int = 0):
        super().__init__()
        self.mingru = MinGRU(d_model)
        self.gin_channels = gin_channels
        if gin_channels > 0:
            self.modulator = nn.Linear(gin_channels, d_model)

    def _initial_state(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.mingru._initial_state(x, *args, **kwargs)

    def _parallel_forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        g: Tensor | None = None,
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        x, h_last = self.mingru._parallel_forward(x, h_prev)
        if self.gin_channels > 0 and g is not None:
            x = x * F.sigmoid(self.modulator(g))
        return x, h_last


@derive_config
class NhvcEncoder(StatefulModule):
    """
    NHVC Encoder, this module encodes phoneme without speaker-specific information, and estimate pitch, noise-gate.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_fft: int = 1024,
        dim_phonemes: int = 64,
        n_f0_classes: int = 128,
    ):
        super().__init__()
        self.fft_bin = n_fft // 2 + 1
        self.pre = nn.Linear(self.fft_bin, d_model)
        self.stack = StatefulModuleSequential(
            [NhvcLayer(d_model) for _ in range(n_layers)]
        )
        self.post = nn.Linear(d_model, dim_phonemes + n_f0_classes + self.fft_bin)

    def _initial_state(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.stack._initial_state(x, *args, **kwargs)

    def _parallel_forward(
        self, x: Tensor, h: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        x = self.pre(x)
        x, h_last = self.stack._parallel_forward(x, h, *args, **kwargs)
        x = self.post(x)
        return x, h_last


@derive_config
class NhvcDecoder(StatefulModule):
    """
    NHVC Decoder, this module estimates parameters of subtractive vocoder
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        dim_phonemes: int = 64,
        n_fft: int = 1024,
        dim_periodicity: int = 16,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.fft_bin = n_fft // 2 + 1
        self.pre = nn.Linear(dim_phonemes, d_model)
        self.stack = StatefulModuleSequential(
            [NhvcLayer(d_model, gin_channels=gin_channels) for _ in range(n_layers)]
        )
        self.post = nn.Linear(d_model, dim_periodicity + self.fft_bin)

    def _initial_state(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.stack._initial_state(x, *args, **kwargs)

    def _parallel_forward(
        self, x: Tensor, h: Tensor, g: Tensor | None = None, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        x = self.pre(x)
        x, h_last = self.stack._parallel_forward(x, h, *args, g=g, **kwargs)
        x = self.post(x)
        return x, h_last
