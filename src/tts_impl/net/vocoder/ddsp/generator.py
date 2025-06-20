from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.base.vocoder import GanVocoderGenerator
from tts_impl.net.common.grux import Grux
from tts_impl.utils.config import derive_config

from .vocoder import SubtractiveVocoder


@derive_config
class DdspGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        in_channels: int = 80,
        d_model: int = 256,
        num_layers: int = 4,
        reverb_size: int = 1024,
        gin_channels: int = 0,
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
    ):
        super().__init__()
        self.fft_bin = vocoder.n_fft // 2 + 1
        out_channels = vocoder.dim_periodicity + vocoder.dim_envelope
        self.reverb_size = reverb_size
        self.sample_rate = vocoder.sample_rate
        self.vocoder = SubtractiveVocoder(**vocoder)
        self.conv_pre = nn.Conv1d(in_channels, d_model, 1)
        self.conv_post = nn.Conv1d(d_model, out_channels, 1)
        self.gin_channels = gin_channels
        self.sample_rate = vocoder.sample_rate

        self.grux = Grux(d_model, num_layers, d_condition=gin_channels)
        if reverb_size > 0:
            self.reverb_noise = nn.Parameter(
                F.normalize(torch.randn(reverb_size), dim=0)[None, :]
            )
            if gin_channels > 0:
                self.to_reverb_parameters = nn.Conv1d(
                    gin_channels, 2, 1
                )  # (decay, wet)
                t = torch.arange(reverb_size).float() / self.sample_rate
                self.register_buffer("t", t)
        else:
            self.reverb_noise = None

    def net(self, x, g=None):
        x = self.conv_pre(x)
        x = x.transpose(1, 2)
        x, _ = self.grux(x, c=g)
        x = x.transpose(1, 2)
        x = self.conv_post(x)
        x = x.float()
        per, env = torch.split(
            x, [self.vocoder.dim_periodicity, self.vocoder.dim_envelope], dim=1
        )
        per = torch.sigmoid(per)
        env = torch.exp(env.float())
        return per, env

    def forward(self, x, f0, g=None, uv=None):
        p, e = self.net(x, g=g)
        x = self.vocoder.synthesize(f0, p, e)
        x = x.unsqueeze(dim=1)
        return x
