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
        d_model: int = 128,
        num_layers: int = 6,
        vocal_cord_size: int = 256,
        reverb_size: int = 2048,
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
    ):
        super().__init__()
        out_channels = vocoder.n_mels + vocoder.dim_periodicity
        self.sample_rate = vocoder.sample_rate
        self.vocoder = SubtractiveVocoder(**vocoder)
        self.conv_pre = nn.Conv1d(in_channels, d_model, 1)
        self.conv_post = nn.Conv1d(d_model, out_channels, 1)
        self.grux = Grux(d_model, num_layers)
        self.vocal_cord = nn.Parameter(
            F.normalize(torch.randn(vocal_cord_size), dim=0)[None, :]
        )
        self.reverb = nn.Parameter(
            F.normalize(torch.randn(reverb_size), dim=0)[None, :]
        )

    def net(self, x):
        x = self.conv_pre(x)
        x = x.transpose(1, 2)
        x, _ = self.grux(x)
        x = x.transpose(1, 2)
        x = self.conv_post(x)
        x = x.float()
        p, e = torch.split(
            x, [self.vocoder.dim_periodicity, self.vocoder.n_mels], dim=1
        )
        p = torch.sigmoid(p)
        e = torch.sigmoid(e)
        return p, e

    def forward(self, x, f0, uv=None):
        p, e = self.net(x)
        v = F.normalize(
            self.vocal_cord.expand(x.shape[0], self.vocal_cord.shape[1]), dim=1
        )
        r = F.normalize(self.reverb.expand(x.shape[0], self.reverb.shape[1]), dim=1)
        x = self.vocoder.forward(f0, p, e, vocal_cord=v, reverb=r)
        x = x.unsqueeze(dim=1)
        return x
