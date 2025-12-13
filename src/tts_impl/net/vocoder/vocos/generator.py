import torch
from torch import nn as nn

from tts_impl.net.base import GanVocoderGenerator
from tts_impl.net.common.convnext import ConvNeXt1d
from tts_impl.net.common.stft import ISTFT
from tts_impl.utils.config import derive_config


@derive_config
class VocosGenerator(nn.Module, GanVocoderGenerator):
    """
    Unofficial implementation of [Vocos](https://arxiv.org/abs/2306.00814)'s generator.
    Using transpose-convolution instead of `torch.istft` for onnx exporting compatibility.
    """

    def __init__(
        self,
        in_channels: int = 80,
        inter_channels: int = 512,
        ffn_channels: int = 1536,
        kernel_size: int = 7,
        n_fft: int = 1024,
        hop_length: int = 256,
        num_layers: int = 8,
        grn: bool = False,
        glu: bool = False,
        norm: str = "layernorm",
        activation: str = "gelu",
    ):
        super().__init__()
        fft_bin = n_fft // 2 + 1
        self.fft_bin = fft_bin
        self.convnext = ConvNeXt1d(
            in_channels,
            fft_bin * 2,
            inter_channels,
            ffn_channels,
            kernel_size,
            num_layers,
            grn,
            glu,
            norm,
            activation,
        )
        self.istft = ISTFT(n_fft, hop_length)

    def forward(self, x):
        x = self.convnext(x)
        x = x.float()
        log_mag, phase = torch.chunk(x, 2, dim=1)
        mag = torch.exp(torch.clamp_max(log_mag, max=6.0))
        real = torch.cos(phase) * mag
        imag = torch.sin(phase) * mag
        output = self.istft.forward(real, imag)
        return output
