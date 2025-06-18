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
        reverb_size: int = 4096,
        gin_channels: int = 0,
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
    ):
        super().__init__()
        out_channels = vocoder.n_mels * 2
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
        env_imp, env_noi = torch.split(
            x, [self.vocoder.n_mels, self.vocoder.n_mels], dim=1
        )
        env_imp = torch.sigmoid(env_imp)
        env_noi = torch.sigmoid(env_noi)
        return env_imp, env_noi

    def _calculate_reverb(
        self, g: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        if g is not None and self.reverb_size > 0:
            decay, wet = self.to_reverb_parameters(g).split(
                [1, 1], dim=1
            )  # [batch_size, 1, 1]
            decay = decay[:, 0]  # [batch_size, 1]
            wet = wet[:, 0]  # [batch_size, 1]
            coeff = torch.exp(
                -F.softplus(-decay) * self.t * 500.0
            )  # [batch_size, reverb_size]
            reverb = self.reverb_noise * coeff * torch.sigmoid(wet)
            reverb = F.normalize(reverb, dim=1)
            return reverb
        elif self.reverb_size > 0:
            reverb = F.normalize(
                self.reverb_noise.expand(batch_size, self.reverb_noise.shape[1]), dim=1
            )
            return reverb
        else:
            return None


    def forward(self, x, f0, g=None, uv=None):
        p, e = self.net(x, g=g)
        r = self._calculate_reverb(g=g, batch_size=x.shape[0])
        x = self.vocoder.forward(f0, p, e, reverb=r)
        x = x.unsqueeze(dim=1)
        return x
