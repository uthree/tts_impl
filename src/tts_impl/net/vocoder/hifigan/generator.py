# HiFi-GAN from https://arxiv.org/abs/2010.05646
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from alias_free_torch import Activation1d as AliasFreeActivation1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from tts_impl.net.base.vocoder import GanVocoderGenerator
from tts_impl.utils.config import derive_config

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class Snake(nn.Module):
    def __init__(
        self,
        channels: int,
        alpha: float = 0.0,
        trainable: bool = True,
        logscale: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.logscale = logscale
        self.eps = eps

        if logscale:
            self.alpha = nn.Parameter(torch.zeros(channels) + alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(channels) * alpha)

        self.alpha.requires_grad = trainable

    def forward(self, x):
        alpha = self.alpha[None, :, None]
        if self.logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.eps)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class SnakeBeta(nn.Module):
    def __init__(
        self,
        channels: int,
        alpha: float = 0.0,
        beta: float = 0.0,
        trainable: bool = True,
        logscale: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.logscale = logscale
        self.eps = eps

        if logscale:
            self.alpha = nn.Parameter(torch.zeros(channels) + alpha)
            self.beta = nn.Parameter(torch.zeros(channels) + beta)
        else:
            self.alpha = nn.Parameter(torch.ones(channels) * alpha)
            self.beta = nn.Parameter(torch.ones(channels) * beta)

        self.alpha.requires_grad = trainable
        self.beta.requires_grad = trainable

    def forward(self, x):
        alpha = self.alpha[None, :, None]
        beta = self.beta[None, :, None]
        if self.logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.eps)) * torch.pow(torch.sin(x * alpha), 2)
        return x


def init_activation(name: str = "lrelu", channels: int = 0, alias_free: bool = False):
    if alias_free:
        act = init_activation(name, channels=channels, alias_free=False)
        act = AliasFreeActivation1d(act)
        return act

    if name == "lrelu":
        return nn.LeakyReLU(LRELU_SLOPE)
    elif name == "silu":
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "snake":
        return Snake(channels=channels)
    elif name == "snakebeta":
        return SnakeBeta(channels=channels)
    else:
        raise ValueError(
            'Invalid activation name. available: ["lrelu", "silu", "gelu", "snake", "snakebeta"]'
        )


class ResBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
        activation: str = "lrelu",
        alias_free: bool = False,
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.acts1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.acts2 = nn.ModuleList()
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
            self.acts1.append(
                init_activation(activation, channels, alias_free=alias_free)
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
            self.acts2.append(
                init_activation(activation, channels, alias_free=alias_free)
            )

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_weight_norm(c1)
            remove_weight_norm(c2)


class ResBlock2(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3],
        activation: str = "lrelu",
        alias_free: bool = False,
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.acts1 = nn.ModuleList()
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
            self.acts1.append(
                init_activation(activation, channels, alias_free=alias_free)
            )

    def forward(self, x):
        for c1, a1 in zip(self.convs1, self.acts1):
            xt = a1(x)
            xt = c1(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c1 in self.convs1:
            remove_weight_norm(c1)


@derive_config
class HifiganGenerator(nn.Module, GanVocoderGenerator):
    """
    HiFi-GAN Generator purposed in https://arxiv.org/abs/2010.05646
    """

    def __init__(
        self,
        in_channels: int = 80,
        upsample_initial_channels: int = 512,
        resblock_type: str = "1",
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates: List[int] = [8, 8, 2, 2],
        out_channels: int = 1,
        tanh_post_activation: bool = True,
        gin_channels: int = 0,
        activation: str = "lrelu",
        alias_free: bool = False,
        sample_rate: int = 22050,
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
        self.upsample_rates = upsample_rates
        self.out_channels = out_channels
        self.tanh_post_activation = tanh_post_activation
        self.gin_channels = gin_channels
        self.sample_rate = sample_rate

        if resblock_type == "1":
            resblock = ResBlock1
        elif resblock_type == "2":
            resblock = ResBlock2
        else:
            raise ValueError("invalid resblock type")

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
        self.up_acts = nn.ModuleList()
        for i, u in enumerate(upsample_rates):
            c1 = upsample_initial_channels // (2**i)
            c2 = upsample_initial_channels // (2 ** (i + 1))
            p = u // 2
            k = u * 2
            self.up_acts.append(
                init_activation(activation, channels=c1, alias_free=alias_free)
            )
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, p)))
            self.frame_size *= u

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilations)):
                self.resblocks.append(resblock(ch, k, d, alias_free=alias_free))

        self.post_act = init_activation(activation, channels=c2, alias_free=alias_free)
        self.conv_post = weight_norm(nn.Conv1d(ch, out_channels, 7, 1, padding=3))

        self.apply(init_weights)

    def forward(
        self, x: torch.Tensor, g: Optional[torch.Tensor] = None, *args, **kwargs
    ):
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
            x = self.up_acts[i](x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.post_act(x)
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
