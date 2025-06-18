import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.base.stateful import StatefulModule
from tts_impl.net.common.grux import Grux
from typing import Optional, Tuple


class NsfgruxFilterModule(StatefulModule):
    def __init__(
        self,
        in_channels: int,
        n_fft: int = 1024,
        frame_size: int = 256,
        d_model: int = 256,
        num_layers: int = 6,
        kernel_size: int = 4,
        d_ffn: Optional[int] = None,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.frame_size = frame_size
        self.gin_channels = gin_channels
        self.pre = nn.Linear(in_channels + n_fft + 2, d_model)
        self.post = nn.Linear(d_model, n_fft + 2)
        self.grux = Grux(
            d_model=d_model,
            num_layers=num_layers,
            kernel_size=kernel_size,
            d_ffn=d_ffn,
            d_condition=gin_channels,
        )

    def _parallel_forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pre(x)
        x, h = self.grux(x, h, c=g)
        x = self.post(x)
        return x, h

    def _initial_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.grux._initial_state(x)


class NsfgruxSourceModule(nn.Module):
    def __init__(self, sample_rate: int, num_harmonics: int, gin_channels: int = 0):
        super().__init__()
        self.gin_channels = gin_channels
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
