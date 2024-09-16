# HiFi-GAN Discriminator from https://arxiv.org/abs/2010.05646

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_padding

from torch.nn.utils.parametrizations import weight_norm, spectral_norm

from typing import List, Tuple

LRELU_SLOPE = 0.1


class PeriodDiscriminator(nn.Module):
    def __init__(
            self,
            period: int = 1,
            kernel_size: int = 5,
            stride: int = 3,
            use_spectral_norm: bool = False,
            channels: int = 32,
            channels_max: int = 1024,
            channels_mul: int = 4,
            num_layers: int = 4,
    ):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList()
        self.convs.append(
            norm_f(
                nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), get_padding(kernel_size, 1))
            )
        )
        c = channels
        for _ in range(num_layers):
            self.convs.append(
                norm_f(
                    nn.Conv2d(c, c * channels_mul, (kernel_size, 1), (stride, 1), get_padding(kernel_size, 1))
                )
            )
            c = c * channels_mul
            c = min(c, channels_max)
        self.post = nn.Conv2d(c, 1, (3, 1), 1, (1, 0))

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
            shapes:
                x: [batch_size, channels, time]
                outputs:
                    x: [batch_size, channels, time]
                    fmap: [batch_size, channels, period, time]
        '''
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
    

class ScaleDiscriminator(nn.Module):
    def __init__(
            self,
            scale: int = 1,
            use_spectral_norm: bool = False
    ):
        super().__init__()
        if scale == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool1d(scale*2, scale, scale)

        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
    


class CombinedDiscriminator(nn.Module):
    '''
    Combined multiple discriminators.
    '''
    def __init__(self, sub_discriminators):
        super().__init__()
        self.sub_discriminators = nn.ModuleList(sub_discriminators)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        fmap = []
        logits = []
        for sd in self.sub_discriminators:
            l, f = sd(x)
            if type(l) is list:
                logits += l
            else:
                logits.append(l)
            fmap += f
        return logits, fmap


class MultiPeriodDiscriminator(CombinedDiscriminator):
    def __init__(
            self,
            periods: List[int] = [2, 3, 5, 7, 11],
            kernel_size: int = 5,
            stride: int = 3,
            use_spectral_norm: bool = False,
            channels: int = 32,
            channels_max: int = 1024,
            channels_mul: int = 4,
            num_layers: int = 4,
    ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList()
        for p in periods:
            self.sub_discriminators.append(PeriodDiscriminator(p, kernel_size, stride, use_spectral_norm, channels, channels_max, channels_mul, num_layers))


class MultiScaleDiscriminator(CombinedDiscriminator):
    def __init__(
            self,
            scales=[1, 2, 4]        
    ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList()
        for i, s in enumerate(scales):
            self.sub_discriminators.append(ScaleDiscriminator(s, i==0))
