from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from tts_impl.net.vocoder.base import GanVocoderGenerator
from tts_impl.net.vocoder.hifigan.generator import (ResBlock1, ResBlock2,
                                                    init_weights)

from .oscillator import HarmonicNoiseOscillator


class NsfhifiganGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        input_channels: int = 80,
        upsample_initial_channels: int = 512,
        resblock_type: Literal["1", "2"] = "1",
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_rates: List[int] = [8, 8, 2, 2],
        output_channels: int = 1,
        tanh_post_activation: bool = True,
        # option for speaker conditioning in TTS task
        condition_channels: int = 0,
        # for source module
        sample_rate: float = 22050.0,
        num_harmonics: int = 1,
    ):
        super().__init__()
        self.requires_f0 = True

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.frame_size = 1

        self.input_channels = input_channels
        self.condition_channels = condition_channels
        self.upsample_initial_channels = upsample_initial_channels
        self.resblock_type = resblock_type
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilations = resblock_dilations
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_rates = upsample_rates
        self.output_channels = output_channels
        self.tanh_post_activation = tanh_post_activation

        if resblock_type == "1":
            resblock = ResBlock1
        elif resblock_type == "2":
            resblock = ResBlock2
        else:
            raise "invalid resblock type"

        self.conv_pre = weight_norm(
            nn.Conv1d(input_channels, upsample_initial_channels, 7, 1, 3)
        )
        if condition_channels > 0:
            self.with_condition = True
            self.conv_cond = weight_norm(
                nn.Conv1d(condition_channels, upsample_initial_channels, 1, bias=False)
            )
        else:
            self.with_condition = False
            self.conv_cond = None

        self.ups = nn.ModuleList()
        downs = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c1 = upsample_initial_channels // (2**i)
            c2 = upsample_initial_channels // (2 ** (i + 1))
            p = (k - u) // 2
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, p)))
            downs.append(weight_norm(nn.Conv1d(c2, c1, k, u, p)))
            self.frame_size *= u
        downs.append(weight_norm(nn.Conv1d(1, c2, 7, 1, 3)))
        self.downs = nn.ModuleList(list(reversed(downs)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilations)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, output_channels, 7, 1, padding=3))

        self.source_module = HarmonicNoiseOscillator(
            sample_rate, self.frame_size, num_harmonics
        )

        self.apply(init_weights)

    def forward(
        self,
        features: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None,
        uv: Optional[torch.Tensor] = None,
    ):
        """
        inputs:
            features: [batch_size, in_channels, num_frames], dtype=float
            condition: [batch_size, 1, condition_dim] dtype=float, optional
            f0: [batch_size, 1, num_frames], dtype=float, optional
            uv: [batch_size, 1, num_frames], dtype=float, optional
        returns:
            waveform: [batch_size, out_channels, frames*frame_size]
                where: frame_size is the number of samples per frame.
        """
        if f0 is None:
            f0 = torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device)
        if uv is None:
            uv = (f0 > 0.0).to(x.dtype)

        s = self.source_module.forward(f0, uv)
        source_signals = []
        for i in range(len(self.downs)):
            s = self.downs[i](s)
            s = F.leaky_relu(s, 0.1)
            source_signals.append(s)
        source_signals = list(reversed(source_signals))

        x = self.conv_pre(features)
        if condition is not None:
            x = x + self.conv_cond(condition)
        for i in range(self.num_upsamples):
            x = x + source_signals[i]
            x = self.ups[i](x)
            x = F.leaky_relu(x, 0.1)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = x + source_signals[-1]
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        if self.tanh_post_activation:
            x = torch.tanh(x)
        return x
