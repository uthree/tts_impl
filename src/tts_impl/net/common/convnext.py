import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from .normalization import LayerNorm1d, GlobalResponseNorm1d, DynamicTanh1d

class DepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, causal=False, norm: Literal["layernorm", "none", "tanh"] = "layernorm"):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )
        self.kernel_size = kernel_size
        if norm == "tanh":
            self.norm = DynamicTanh1d(channels)
        elif norm == "none":
            self.norm = nn.Identity()
        elif norm  == "layernorm":
            self.norm = LayerNorm1d(channels)
        else:
            raise RuntimeError("invalid norm")

        if causal:
            self.pad = nn.ReflectionPad1d((0, self.kernel_size - 1))
        else:
            self.pad = nn.ReflectionPad1d(
                (self.kernel_size // 2, self.kernel_size // 2)
            )
        self.state_length = kernel_size - 1

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        return x
    

class FeedForward1d(nn.Module):
    def __init__(self, channels: int, internal_channels: int, grn=True, glu=False):
        super().__init__()
        self.glu = glu
        if glu:
            self.c1 = nn.Conv1d(channels, internal_channels * 2, 1)
        else:
            self.c1 = nn.Conv1d(channels, internal_channels, 1)
        self.c2 = nn.Conv1d(internal_channels, channels, 1)
        if grn:
            self.grn = GlobalResponseNorm1d(internal_channels)
        else:
            self.grn = nn.Identity()

    def forward(self, x):
        if self.glu:
            x = self.c1(x)
            x_0, x_1 = x.chunk(2, dim=1)
            x = x_0 * F.gelu(x_1)
        else:
            x = self.c1(x)
            x = F.gelu(x)
        x = self.grn(x)
        x = self.c2(x)
        return x
    

class ConvNeXtLayer1d(nn.Module):
    def __init__(
        self,
        channels,
        ffn_channels,
        kernel_size=7,
        grn=True,
        glu=False,
        norm="layernorm",
        causal=True,
    ):
        super().__init__()
        self.dw = DepthwiseConv1d(channels, kernel_size, causal, norm)
        self.ffn = FeedForward1d(channels, ffn_channels, grn, glu)

    def forward(self, x):
        res = x
        x = self.dw(x)
        x = self.ffn(x)
        x = res + x
        return x


class ConvNeXt1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inter_channels: int = 256,
        ffn_channels: int = 512,
        kernel_size: int = 7,
        num_layers: int = 6,
        grn: bool = True,
        glu: bool = True,
        norm: str = "layernorm",
        causal: bool = False,
    ):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, inter_channels, 1)
        self.layers = nn.ModuleList(
            [
                ConvNeXtLayer1d(
                    inter_channels,
                    ffn_channels,
                    kernel_size,
                    grn,
                    glu,
                    norm,
                    causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_norm = LayerNorm1d(inter_channels) if norm else nn.Identity()
        self.out_conv = nn.Conv1d(inter_channels, out_channels, 1)

    def forward(self, x):
        x = self.in_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.post_norm(x)
        x = self.out_conv(x)
        return x