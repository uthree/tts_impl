import torch
from torch import nn as nn

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
        vocoder: HomomorphicVocoder.Config = HomomorphicVocoder.Config(),
    ):
        super().__init__()
        self.sample_rate = vocoder.sample_rate
        self.vocoder = HomomorphicVocoder(**vocoder)
        self.gin_channels = gin_channels
        self.sample_rate = vocoder.sample_rate
        self.fft_bin = vocoder.n_fft // 2 + 1
        self.pre = nn.Conv1d(in_channels, d_model, 1)
        self.wn = WN(d_model, 3, 1, num_layers, gin_channels)
        self.post = nn.Conv1d(d_model, self.fft_bin * 2, 1)

    def net(self, x, g=None):
        x = self.pre(x)
        x_mask = torch.ones(x.shape[0], 1, x.shape[2], device=x.device)
        x = self.wn(x, x_mask, g=g)
        x = self.post(x)
        x = x.float()
        per, noi = torch.split(x, [self.fft_bin, self.fft_bin], dim=1)
        per = torch.exp(per)
        noi = torch.exp(noi)
        return per, noi

    def forward(self, x, f0, g=None, uv=None):
        per, noi = self.net(x, g=g)
        x = self.vocoder.forward(f0, per, noi)
        x = x.unsqueeze(dim=1)
        return x
