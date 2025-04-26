from typing import Optional

import torch
import torch.nn as nn
from tts_impl.net.base.vocoder import GanVocoderGenerator
from tts_impl.utils.config import derive_config

from .filter import NsfhifiganFilter
from .oscillator import HarmonicNoiseOscillator


@derive_config
class NsfhifiganGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        source: HarmonicNoiseOscillator.Config = HarmonicNoiseOscillator.Config(),
        filter: NsfhifiganFilter.Config = NsfhifiganFilter.Config(),
    ):
        super().__init__()
        self.source_module = HarmonicNoiseOscillator(**source)
        self.filter = NsfhifiganFilter(**filter)

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

        if f0 is None:
            f0 = torch.zeros((x.shape[0], x.shape[2]), device=x.device)
        if uv is None:
            uv = (f0 > 0.0).to(x.dtype)

        f0 = f0.unsqueeze(1)
        uv = uv.unsqueeze(1)

        s = self.source_module(f0, uv, g=g)
        out = self.filter(x, s, g=g)
        return out
