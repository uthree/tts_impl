from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from tts_impl.net.base.vocoder import GanVocoderGenerator
from tts_impl.net.vocoder.hifigan.generator import ResBlock1, ResBlock2, init_weights
from tts_impl.utils.config import derive_config

from .oscillator import HarmonicNoiseOscillator


@derive_config
class NsfhifiganGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        input_channels: int = 80,
        upsample_initial_channels: int = 512,
        resblock_type: str = "1",
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_rates: List[int] = [8, 8, 2, 2],
        output_channels: int = 1,
        tanh_post_activation: bool = True,
        # option for speaker conditioning in TTS task
        gin_channels: int = 0,
        # for source module
        sample_rate: int = 22050,
        num_harmonics: int = 8,
    ):
        super().__init__()
        self.requires_f0 = True

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.input_channels = input_channels
        self.gin_channels = gin_channels
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
        if gin_channels > 0:
            self.with_condition = True
            self.conv_cond = weight_norm(
                nn.Conv1d(gin_channels, upsample_initial_channels, 1, bias=False)
            )
        else:
            self.with_condition = False
            self.conv_cond = None

        self.frame_size = int(np.prod(upsample_rates))

        self.ups = nn.ModuleList()
        self.source_convs = nn.ModuleList()
        self.source_convs.append(
            weight_norm(
                nn.Conv1d(
                    1,
                    self.upsample_initial_channels,
                    self.frame_size * 2,
                    self.frame_size,
                    self.frame_size // 2,
                )
            )
        )
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c1 = upsample_initial_channels // (2**i)
            c2 = upsample_initial_channels // (2 ** (i + 1))
            pad = (k - u) // 2
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, pad)))
            prod = int(np.prod(upsample_rates[(i + 1) :]))
            if prod != 1:
                self.source_convs.append(
                    weight_norm(nn.Conv1d(1, c2, prod * 2, prod, prod // 2))
                )
            else:
                self.source_convs.append(weight_norm(nn.Conv1d(1, c2, 7, 1, 3)))

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
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None,
        uv: Optional[torch.Tensor] = None,
    ):
        """
        args:
            x: [batch_size, in_channels, num_frames], dtype=float
            g: [batch_size, 1, condition_dim] dtype=float, optional
            f0: [batch_size, num_frames], dtype=float, optional
            uv: [batch_size, num_frames], dtype=float, optional
        outputs:
            waveform: [batch_size, out_channels, frames*frame_size]
                where: frame_size is the number of samples per frame.
        """
        if f0 is None:
            f0 = torch.zeros((x.shape[0], x.shape[2]), device=x.device)
        if uv is None:
            uv = (f0 > 0.0).to(x.dtype)

        f0 = f0.unsqueeze(1)
        uv = uv.unsqueeze(1)

        s = self.source_module.forward(f0, uv)

        x = self.conv_pre(x)
        if g is not None:
            x = x + self.conv_cond(g)

        x + self.source_convs[0](s)

        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            x = F.leaky_relu(x, 0.1)
            x = x + self.source_convs[i + 1](s)
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
