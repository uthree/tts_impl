from torch import nn as nn

from tts_impl.net.base import GanVocoderGenerator
from tts_impl.utils.config import derive_config


# WIP
@derive_config
class WaveHaxGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        in_channels: int = 100,
        internal_channels: int = 32,
        ffn_channels: int = 64,
        n_fft: int = 480,
        hop_length: int = 240,
        kernel_size: int = 7,
        num_harmonics: int = 8,
    ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.n_fft = n_fft
        self.fft_bin = n_fft // 2 + 1
        self.hop_length = hop_length
