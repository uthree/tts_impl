from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from _pytest.config.argparsing import ArgumentError
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm


class IstftSynthesis(nn.Module):
    def __init__(self, in_channels: int, upscale_factor: int = 4):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.n_fft = upscale_factor * 4
        self.hop_length = upscale_factor
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.proj = nn.Conv1d(in_channels, self.n_fft + 2, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        mag, phase = torch.chunk(x, 2, dim=1)
        mag = torch.clamp_max(mag, 4.0)
        amp = torch.exp(mag)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        z = torch.complex(real, imag)
        waveform = torch.istft(
            z, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window
        )
        waveform = waveform.unsqueeze(1)
        waveform = F.pad(waveform, (self.upscale_factor // 2, self.upscale_factor // 2))
        return waveform


class MultiBandSynthesis(nn.Module):
    pass


class MultiStreamSynthesis(nn.Module):
    pass


def init_post_module(
    in_channels: int,
    out_channels: int = 1,
    upscale_factor: int = 1,
    post_module_type: Literal["conv", "istft"] = "conv",
) -> nn.Module:
    if post_module_type == "conv":
        assert upscale_factor == 1, "postmodule's upscale factor must be 1."
        return weight_norm(nn.Conv1d(in_channels, out_channels, 7, 1, padding=3))
    elif post_module_type == "istft":
        assert out_channels == 1, "out_channels must be 1."
        return IstftSynthesis(in_channels, upscale_factor)
    else:
        raise RuntimeError(
            f"{post_module_type} must be one of the following: `conv`, `istft`"
        )
