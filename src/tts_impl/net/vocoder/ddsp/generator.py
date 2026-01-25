import torch
from torch import nn as nn

from tts_impl.net.base.vocoder import GanVocoderGenerator
from tts_impl.net.common.convnext import ConvNeXt1d
from tts_impl.utils.config import derive_config

from .vocoder import HomomorphicVocoder


@derive_config
class DdspGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        in_channels: int = 80,
        d_model: int = 256,
        n_layers: int = 3,
        gin_channels: int = 0,
        vocoder: HomomorphicVocoder.Config = HomomorphicVocoder.Config(),
    ):
        super().__init__()
        self.sample_rate = vocoder.sample_rate
        self.vocoder = HomomorphicVocoder(**vocoder)
        self.gin_channels = gin_channels
        self.sample_rate = vocoder.sample_rate
        self.fft_bin = vocoder.n_fft // 2 + 1
        self.d_periodicity = vocoder.d_periodicity
        self.d_spectral_envelope = vocoder.d_spectral_envelope
        out_channels = self.d_periodicity + self.d_spectral_envelope
        self.convnext = ConvNeXt1d(  # 実はこれ、フレームごとのNHVパラメータ推定ならただのMLPで事足りるのでは？
            in_channels,
            out_channels,
            d_model,
            n_layers=n_layers,
            grn=True,
            glu=True,
        )

    def net(self, x, g=None):
        x = self.convnext(x)
        per, env = torch.split(x, [self.d_periodicity, self.d_spectral_envelope], dim=1)
        per = torch.sigmoid(per)
        env = torch.exp(env)
        return per, env

    def forward(self, x, f0, g=None, uv=None):
        per, env = self.net(x, g=g)
        x = self.vocoder.forward(f0, per, env)
        x = x.unsqueeze(dim=1)
        return x
