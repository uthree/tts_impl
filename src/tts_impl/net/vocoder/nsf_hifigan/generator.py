from typing import Optional

import torch
from torch import nn as nn
from tts_impl.net.base.vocoder import GanVocoderGenerator
from tts_impl.utils.config import derive_config

from .filter import NsfhifiganFilter
from .oscillator import HarmonicNoiseOscillator


@derive_config
class NsfhifiganGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        source_module: HarmonicNoiseOscillator.Config = HarmonicNoiseOscillator.Config(),
        filter_module: NsfhifiganFilter.Config = NsfhifiganFilter.Config(),
    ):
        super().__init__()
        self.filter_module = NsfhifiganFilter(**filter_module)
        self.source_module = HarmonicNoiseOscillator(**source_module)

    @property
    def sample_rate(self) -> int:
        return self.source_module.sample_rate

    @property
    def frame_size(self) -> int:
        return self.source_module.frame_size

    def forward(
        self,
        x,
        g: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None,
        uv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: shape=[batch_size, in_channels, num_frames]
            g: shape=[batch_size, gin_channels, 1]
            f0: shape=[batch, num_frames]
            uv: shape=[batch, num_frames]
        """
        assert f0 is not None, RuntimeError("f0 shoud be given")
        if uv is None:
            uv = (f0 >= 20.0).to(x.dtype) * (f0 <= (self.sample_rate / 2)).to(x.dtype)

        s = self.source_module(f0.unsqueeze(1), uv.unsqueeze(1), g=g)
        out = self.filter_module(x, s, g=g)
        return out
