# HiFi-GAN from https://arxiv.org/abs/2010.05646
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from tts_impl.net.base.vocoder import GanVocoderGenerator

from dataclasses import dataclass, field


LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilations: List[int] = [1, 3, 5]
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        get_padding(kernel_size, d),
                        d,
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        get_padding(kernel_size, 1),
                        1,
                    )
                )
            )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_weight_norm(c1)
            remove_weight_norm(c2)


class ResBlock2(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilations: List[int] = [1, 3]
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        get_padding(kernel_size, d),
                        d,
                    )
                )
            )

    def forward(self, x):
        for c1 in self.convs1:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c1 in self.convs1:
            remove_weight_norm(c1)


@dataclass
class HifiganGeneratorConfig:
    """
    hyperparameters of HiFi-GAN
    """

    in_channels: int = 80
    upsample_initial_channels: int = 512
    resblock_type: Literal["1", "2"] = "1"
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilations: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    out_channels: int = 1
    tanh_post_activation: bool = True
    gin_channels: int = 0


class HifiganGenerator(nn.Module, GanVocoderGenerator):
    """
    HiFi-GAN Generator purposed in https://arxiv.org/abs/2010.05646
    """

    def __init__(
        self,
        in_channels: int = 80,
        upsample_initial_channels: int = 512,
        resblock_type: Literal["1", "2"] = "1",
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_rates: List[int] = [8, 8, 2, 2],
        out_channels: int = 1,
        tanh_post_activation: bool = True,
        gin_channels: int = 0,
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.frame_size = 1

        self.in_channels = in_channels
        self.upsample_initial_channels = upsample_initial_channels
        self.resblock_type = resblock_type
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilations = resblock_dilations
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_rates = upsample_rates
        self.out_channels = out_channels
        self.tanh_post_activation = tanh_post_activation
        self.gin_channels = gin_channels

        if resblock_type == "1":
            resblock = ResBlock1
        elif resblock_type == "2":
            resblock = ResBlock2
        else:
            raise "invalid resblock type"

        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channels, 7, 1, 3)
        )
        if gin_channels > 0:
            self.with_condition = True
            self.conv_cond = weight_norm(
                nn.Conv1d(gin_channels, upsample_initial_channels, 1, bias=False)
            )
        else:
            self.with_condition = False
            self.conv_cond = None

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c1 = upsample_initial_channels // (2**i)
            c2 = upsample_initial_channels // (2 ** (i + 1))
            p = (k - u) // 2
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, p)))
            self.frame_size *= u

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilations)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, out_channels, 7, 1, padding=3))

        self.apply(init_weights)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        """
        inputs:
            x: [batch_size, in_channels, num_frames], dtype=float
            g: [batch_size, 1, condition_dim] dtype=float, optional
        returns:
            waveform: [batch_size, out_channels, frames*frame_size]
                where: frame_size is the number of samples per frame.
        """
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.conv_cond(g)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            x = F.leaky_relu(x, 0.1)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        if self.tanh_post_activation:
            x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for b in self.resblocks:
            b.remove_weight_norm()
        if self.with_condition:
            remove_weight_norm(self.conv_cond)
