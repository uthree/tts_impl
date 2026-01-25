from typing import Literal

from torch import nn as nn

from tts_impl.utils.config import derive_config

from .activation import ActivationName, init_activation
from .normalization import DynamicTanh1d, GlobalResponseNorm1d, LayerNorm1d


def init_norm(norm: str, channels: int = 0) -> nn.Module:
    if norm == "tanh":
        return DynamicTanh1d(channels)
    elif norm == "none":
        return nn.Identity()
    elif norm == "layernorm":
        return LayerNorm1d(channels)
    else:
        raise RuntimeError("invalid norm")


class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        causal=False,
        norm: Literal["layernorm", "none", "tanh"] = "layernorm",
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )
        self.kernel_size = kernel_size
        self.norm = init_norm(norm, channels)

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
    def __init__(
        self,
        channels: int,
        internal_channels: int,
        grn=True,
        glu=False,
        activation: ActivationName = "gelu",
    ):
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

        self.act = init_activation(activation, channels=internal_channels)

    def forward(self, x):
        if self.glu:
            x = self.c1(x)
            x_0, x_1 = x.chunk(2, dim=1)
            x = x_0 * self.act(x_1)
        else:
            x = self.c1(x)
            x = self.act(x)
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
        activation: ActivationName = "gelu",
    ):
        super().__init__()
        self.dw = DepthwiseConv1d(channels, kernel_size, causal, norm)
        self.ffn = FeedForward1d(channels, ffn_channels, grn, glu, activation)

    def forward(self, x):
        res = x
        x = self.dw(x)
        x = self.ffn(x)
        x = res + x
        return x


@derive_config
class ConvNeXt1d(nn.Module):
    """
    1-dimensional version of [ConvNeXt](https://arxiv.org/abs/2201.03545).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inter_channels: int = 512,
        ffn_channels: int = 1536,
        kernel_size: int = 7,
        n_layers: int = 6,
        grn: bool = False,
        glu: bool = False,
        norm: str = "layernorm",
        activation: ActivationName = "gelu",
        causal: bool = False,
    ):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            inter_channels: number of internal channels
            ffn_channels: number of feed-forward network's internal channels
            kernel_size: length of kernel
            n_layers: number of layers
            grn: if True is given, use Global Response Normalization, purposed at [ConvNeXt V2](https://arxiv.org/abs/2301.00808)
            glu: if true is given, use Gated Linear Units, purposed at [GLU Variants Improve Transformers](https://arxiv.org/abs/2002.05202)
            norm: "layernorm", "tanh" or "none", if "tanh" given, use [DynamicTanh](https://arxiv.org/abs/2503.10622)., default is "layernorm".
            causal: if True is given, this model doesn't refer past information, this option is useful for streaming inference model.
        """
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
                    activation,
                )
                for _ in range(n_layers)
            ]
        )
        self.post_norm = init_norm(norm, inter_channels)
        self.out_conv = nn.Conv1d(inter_channels, out_channels, 1)

    def forward(self, x):
        """
        Args:
            x: shape=(batch_size, in_channels, length)

        Returns:
            x: shape=(batch_size, out_channels, length)
        """
        x = self.in_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.post_norm(x)
        x = self.out_conv(x)
        return x
