import torch
import torch.nn as nn
import torch.nn.functional as F

from tts_impl.net.base import GanVocoderGenerator
from tts_impl.net.common.convnext import ConvNeXt1d
from tts_impl.net.common.normalization import LayerNorm1d
from tts_impl.net.common.stft import ISTFT, STFT
from tts_impl.utils.config import derive_config


@derive_config
class VocosGenerator(nn.Module, GanVocoderGenerator):
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
        log_mag, phase = torch.chunk(x, 2, dim=1)
        mag = torch.exp(log_mag)
        real = torch.cos(phase) * mag
        imag = torch.sin(phase) * mag
        output = self.istft.forward(real, imag)
        return output
