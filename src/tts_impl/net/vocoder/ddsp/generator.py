import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.utils.config import derive_config

from .vocoder import SubtractiveVocoder


# とりあえず仮実装
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, 3, 1, 1)
        self.c2 = nn.Conv1d(channels, channels, 3, 1, 1)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = F.gelu(x)
        x = self.c2(x)
        x = x + res
        return x


@derive_config
class DdspGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        channels: int = 512,
        num_layers: int = 4,
        vocal_cord_size: int = 256,
        reverb_size: int = 2048,
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
    ):
        super().__init__()
        out_channels = vocoder.n_mels + vocoder.dim_periodicity
        self.sample_rate = vocoder.sample_rate
        self.vocoder = SubtractiveVocoder(**vocoder)
        self.input_layer = nn.Conv1d(in_channels, channels, 1)
        self.mid_layers = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_layers)]
        )
        self.output_layer = nn.Conv1d(channels, out_channels, 1)
        self.vocal_cord = nn.Parameter(F.normalize(torch.randn(vocal_cord_size), dim=0)[None, :])
        self.reverb = nn.Parameter(F.normalize(torch.randn(reverb_size), dim=0)[None, :])

    def net(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        p, e = torch.split(x, [self.vocoder.dim_periodicity, self.vocoder.n_mels], dim=1)
        p = torch.sigmoid(p)
        e = F.softplus(e)
        return p, e

    def forward(self, x, f0, uv=None):
        p, e = self.net(x)
        v = F.normalize(self.vocal_cord.expand(x.shape[0], self.vocal_cord.shape[1]), dim=1)
        r = F.normalize(self.reverb.expand(x.shape[0], self.reverb.shape[1]), dim=1)
        x = self.vocoder.forward(f0, p, e, vocal_cord=v, post_filter=r)
        x = x.unsqueeze(dim=1)
        return x
