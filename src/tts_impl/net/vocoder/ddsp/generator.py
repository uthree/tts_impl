import math

import torch
import torchaudio
from torch import nn as nn
from torch.nn import functional as F
from tts_impl.net.base.vocoder import GanVocoderGenerator
from tts_impl.net.tts.vits.modules import WN
from tts_impl.utils.config import derive_config

from .vocoder import HomomorphicVocoder


@derive_config
class DdspGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        in_channels: int = 80,
        d_model: int = 256,
        num_layers: int = 3,
        gin_channels: int = 0,
        dim_periodicity: int = 16,
        vocoder: HomomorphicVocoder.Config = HomomorphicVocoder.Config(),
    ):
        super().__init__()
        self.sample_rate = vocoder.sample_rate
        self.vocoder = HomomorphicVocoder(**vocoder)
        self.dim_periodicity = dim_periodicity
        self.gin_channels = gin_channels
        self.sample_rate = vocoder.sample_rate
        self.fft_bin = vocoder.n_fft // 2 + 1
        self.per_inv_mel = torchaudio.transforms.InverseMelScale(
            self.fft_bin, dim_periodicity, self.vocoder.sample_rate
        )
        self.pre = nn.Conv1d(in_channels, d_model, 1)
        self.wn = WN(d_model, 3, 1, num_layers, gin_channels)
        self.post = nn.Conv1d(d_model, self.dim_periodicity + self.fft_bin, 1)

    def net(self, x, g=None):
        x = self.pre(x)
        x_mask = torch.ones(x.shape[0], 1, x.shape[2], device=x.device)
        x = self.wn(x, x_mask, g=g)
        x = self.post(x)
        x = x.float()
        per, env = torch.split(x, [self.dim_periodicity, self.fft_bin], dim=1)
        per = self.per_inv_mel(per)
        per = torch.sigmoid(per)
        env = torch.exp(env)
        return per, env

    def forward(self, x, f0, g=None, uv=None):
        per, env = self.net(x, g=g)
        x = self.vocoder.forward(f0, per, env)
        x = x.unsqueeze(dim=1)
        return x
