import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.base.vocoder import GanVocoderGenerator
from tts_impl.net.common.convnext import ConvNeXt1d
from tts_impl.utils.config import derive_config

from .vocoder import SubtractiveVocoder


@derive_config
class DdspGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        in_channels: int = 80,
        inter_channels: int = 256,
        num_layers: int = 4,
        kernel_size: int = 7,
        ffn_mul: int = 2,
        grn: bool = False,
        glu: bool = False,
        vocal_cord_size: int = 256,
        reverb_size: int = 2048,
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
    ):
        super().__init__()
        out_channels = vocoder.n_mels + vocoder.dim_periodicity
        self.sample_rate = vocoder.sample_rate
        self.vocoder = SubtractiveVocoder(**vocoder)
        self.convnext = ConvNeXt1d(
            in_channels,
            out_channels,
            inter_channels,
            inter_channels * ffn_mul,
            kernel_size,
            num_layers,
            grn,
            glu,
        )
        self.vocal_cord = nn.Parameter(
            F.normalize(torch.randn(vocal_cord_size), dim=0)[None, :]
        )
        self.reverb = nn.Parameter(
            F.normalize(torch.randn(reverb_size), dim=0)[None, :]
        )

    def net(self, x):
        x = self.convnext(x).float()
        p, e = torch.split(
            x, [self.vocoder.dim_periodicity, self.vocoder.n_mels], dim=1
        )
        p = torch.sigmoid(p)
        e = torch.exp(torch.clamp_max(e, max=6.0))
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
