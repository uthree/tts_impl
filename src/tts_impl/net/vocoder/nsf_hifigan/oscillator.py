import math
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


# HnNSF Module from https://arxiv.org/pdf/1904.12088
class HarmonicNoiseOscillator(nn.Module):
    def __init__(self, sample_rate, frame_size, num_harmonics=1, noise_scale=0.03):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.num_harmonics = num_harmonics
        self.noise_scale = noise_scale

        self.weights = nn.Parameter(torch.zeros(1, num_harmonics, 1))
        self.phase = nn.Parameter(torch.rand(1, num_harmonics, 1))

    def forward(self, f0, uv):
        """
        Args:
            f0: fundamental frequency shape=[N, 1, L]
            uv: unvoiced=0 / voiced=1 flag, shape=[N, 1, L]
        Output
            shape=[N, 1, L * frame_size]
        """
        with torch.no_grad():
            f0 = F.interpolate(f0, scale_factor=self.frame_size, mode="linear")
            voiced_mask = F.interpolate(uv, scale_factor=self.frame_size)
            mul = (
                (torch.arange(self.num_harmonics, device=f0.device) + 1)
                .unsqueeze(0)
                .unsqueeze(2)
            )
            fs = f0 * mul
            integrated = torch.cumsum(fs / self.sample_rate, dim=2)
            rad = 2 * math.pi * ((integrated + self.phase) % 1)
            noise = torch.randn_like(rad)
            harmonics = torch.sin(rad) * voiced_mask + noise * self.noise_scale
            voiced_part = harmonics + noise * self.noise_scale
            unvoiced_part = noise * 0.333
            source = voiced_part * voiced_mask + unvoiced_part * (1 - voiced_mask)
            source = (source * torch.exp(self.weights)).sum(dim=1, keepdim=True)
        return source


class ImpulseOscillator(nn.Module):
    def __init__(
        self,
        sample_rate,
        frame_size,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size

    def forward(self, f0, uv):
        """
        Args:
            f0: fundamental frequency shape=[N, 1, L]
            uv: unvoiced=0 / voiced=1 flag, shape=[N, 1, L]
        Output
            shape=[N, 1, L * frame_size]
        """
        with torch.no_grad():
            f0 = F.interpolate(f0, scale_factor=self.frame_size, mode="linear")
            voiced_mask = F.interpolate(uv, scale_factor=self.frame_size)
            rad = torch.cumsum(-f0 / self.sample_rate, dim=2)
            sawtooth = rad % 1.0
            impluse = sawtooth - sawtooth.roll(1, dims=(2))
            noise = torch.randn_like(impluse) * 0.333
            source = impluse * voiced_mask + noise * (1 - voiced_mask)
        return source


# Cyclic noise oscillator from https://arxiv.org/abs/2004.02191
class CyclicNoiseOscillator(nn.Module):
    def __init__(self, sample_rate, frame_size, base_frequency=440.0, beta=0.78):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.base_frequency = base_frequency
        self.beta = beta

        self.kernel_size = int(4.6 * self.sample_rate / self.base_frequency)
        self.pad_size = self.kernel_size - 1

    def generate_kernel(self):
        t = torch.arange(0, self.kernel_size)[None, None, :]
        decay = torch.exp(-t * self.base_frequency / self.beta / self.sample_rate)
        noise = torch.randn_like(decay)
        kernel = noise * decay
        return kernel

    def forward(self, f0, uv):
        """
        Args:
            f0: fundamental frequency shape=[N, 1, L]
            uv: unvoiced=0 / voiced=1 flag, shape=[N, 1, L]
        Output
            shape=[N, 1, L * frame_size]
        """
        with torch.no_grad():
            f0 = F.interpolate(f0, scale_factor=self.frame_size, mode="linear")
            N = f0.shape[0]
            L = f0.shape[2]
            voiced_mask = F.interpolate(uv, scale_factor=self.frame_size)
            rad = torch.cumsum(-f0 / self.sample_rate, dim=2)
            sawtooth = rad % 1.0
            impluse = sawtooth - sawtooth.roll(1, dims=(2))
            noise = torch.randn(N, 1, L, device=f0.device)
            kernel = self.generate_kernel().to(f0.device)
            impluse = F.pad(impluse, (0, self.pad_size))
            cyclic_noise = F.conv1d(impluse, kernel)
            source = cyclic_noise * voiced_mask + (1 - voiced_mask) * noise
        return source
