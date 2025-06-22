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
        reverb_size: int = 8192,
        gin_channels: int = 0,
        num_reverb_buffers: int = 8,
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
    ):
        super().__init__()
        self.fft_bin = vocoder.n_fft // 2 + 1
        self.reverb_size = reverb_size
        self.sample_rate = vocoder.sample_rate
        self.vocoder = SubtractiveVocoder(**vocoder)
        self.conv_pre = nn.Conv1d(in_channels, d_model, 1)
        self.conv_post = nn.Conv1d(d_model, self.fft_bin * 2, 1)
        self.gin_channels = gin_channels
        self.sample_rate = vocoder.sample_rate

        self.grux = Grux(d_model, num_layers, d_condition=gin_channels)
        if reverb_size > 0:
            self.reverb_noise = nn.Parameter(
                F.normalize(torch.randn(reverb_size), dim=0)[None, :]
            )
            self.reverb_params = nn.Parameter(torch.zeros(1, 2))
            if gin_channels > 0:
                self.to_reverb_noise = nn.Sequential(
                    nn.Conv1d(gin_channels, num_reverb_buffers, 1),
                    nn.Conv1d(num_reverb_buffers, reverb_size, 1, bias=False),
                )
                self.to_reverb_params = nn.Conv1d(gin_channels, 2, 1)
                t = torch.arange(self.reverb_size)[None, :] / self.sample_rate
                self.register_buffer("t", t)

    def net(self, x, g=None):
        x = self.conv_pre(x)
        x = x.transpose(1, 2)
        x, _ = self.grux(x, c=g)
        x = x.transpose(1, 2)
        x = self.conv_post(x)
        x = x.float()
        ir, ap = torch.split(
            x, [self.fft_bin, self.fft_bin], dim=1
        )
        ir = F.softplus(ir)
        ap = F.softplus(ap)
        return ir, ap

    def build_reverb(self, g=None):
        if self.reverb_size > 0:
            # get impulse response
            if self.gin_channels > 0:
                ir = F.normalize(self.to_reverb_noise(g).squeeze(2), dim=1, p=2.0)
            else:
                ir = F.normalize(self.reverb_noise, dim=1, p=2.0)

            # get params
            if self.gin_channels > 0:
                wet, decay = self.to_reverb_params(g).split([1, 1], dim=1)
                wet = wet[:, 0]
                decay = decay[:, 0]
            else:
                wet = self.reverb_params[:, 0]
                decay = self.reverb_params[:, 1]

            # apply wet and decay
            coeff = torch.exp(-F.softplus(-decay) * self.t * 500.0) * torch.sigmoid(wet)
            ir = ir * coeff

            # normalize
            ir = F.normalize(ir, dim=1)
            return ir
        else:
            return None

    def forward(self, x, f0, g=None, uv=None):
        rev = self.build_reverb(g)
        ir, ap = self.net(x, g=g)
        x = self.vocoder.synthesize(f0, ir, ap, reverb=rev)
        x = x.unsqueeze(dim=1)
        return x
