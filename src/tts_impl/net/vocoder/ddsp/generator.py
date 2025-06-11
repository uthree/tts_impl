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
        vocal_cord_size: int = 256,
        reverb_size: int = 4096,
        gin_channels: int = 0,
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
    ):
        super().__init__()
        out_channels = vocoder.n_mels + vocoder.dim_periodicity
        self.reverb_size = reverb_size
        self.vocal_cord_size = vocal_cord_size
        self.sample_rate = vocoder.sample_rate
        self.vocoder = SubtractiveVocoder(**vocoder)
        self.conv_pre = nn.Conv1d(in_channels, d_model, 1)
        self.conv_post = nn.Conv1d(d_model, out_channels, 1)
        self.gin_channels = gin_channels
        self.sample_rate = vocoder.sample_rate

        self.grux = Grux(d_model, num_layers, d_condition=gin_channels)
        if vocal_cord_size > 0:
            if gin_channels > 0:
                self.to_vocal_cord = nn.Conv1d(gin_channels, vocal_cord_size, 1)
            else:
                self.vocal_cord = nn.Parameter(torch.randn(vocal_cord_size)[None, :])
        else:
            self.vocal_cord = None
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
        p, e = torch.split(
            x, [self.vocoder.dim_periodicity, self.vocoder.n_mels], dim=1
        )
        p = torch.sigmoid(p)
        e = torch.sigmoid(e)
        return p, e

    def _calculate_reverb(
        self, g: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        if g is not None and self.gin_channels > 0:
            decay, wet = self.to_reverb_parameters(g).split(
                [1, 1], dim=1
            )  # [batch_size, 1, 1]
            decay = decay[:, 0]  # [batch_size, 1]
            wet = wet[:, 0]  # [batch_size, 1]
            coeff = torch.exp(
                -F.softplus(-decay) * self.t * 500.0
            )  # [batch_size, reverb_size]
            reverb = self.reverb_noise * coeff * torch.sigmoid(wet)
            reverb[:, 0] = 1.0
            return reverb
        elif self.reverb_size > 0:
            return F.normalize(
                self.reverb_noise.expand(batch_size, self.reverb_noise.shape[1]), dim=1
            )
        else:
            return None

    def _calculate_vocal_cord(
        self, g: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        # calculate vocal cord parameter
        if g is not None and self.gin_channels > 0:
            v = F.normalize(
                self.to_vocal_cord(g).squeeze(2), dim=1
            )  # [batch_size, vcord_size]
        elif self.gin_channels <= 0:
            v = F.normalize(
                self.vocal_cord.expand(batch_size, self.vocal_cord.shape[1]), dim=1
            )
        else:
            v = None
        return v

    def forward(self, x, f0, g=None, uv=None):
        p, e = self.net(x, g=g)
        v = self._calculate_vocal_cord(g=g, batch_size=x.shape[0])
        r = self._calculate_reverb(g=g, batch_size=x.shape[0])
        x = self.vocoder.forward(f0, p, e, vocal_cord=v, reverb=r)
        x = x.unsqueeze(dim=1)
        return x
