import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.utils.config import derive_config


# HnNSF Module from https://arxiv.org/pdf/1904.12088
@derive_config
class HarmonicNoiseOscillator(nn.Module):
    def __init__(
        self,
        sample_rate: int = 22050,
        frame_size: int = 256,
        num_harmonics: int = 1,
        noise_scale: float = 0.003,
        gin_channels: int = 0,
        normalize_amps: bool = True,
        post_tanh_activation: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.num_harmonics = num_harmonics
        self.noise_scale = noise_scale
        self.normalize_amps = normalize_amps
        self.post_tanh_activation = post_tanh_activation

        if gin_channels > 0:
            self.take_condition = True
            self.cond_proj = nn.Conv1d(gin_channels, num_harmonics, 1)
            with torch.no_grad():
                self.cond_proj.weight.zero_()
                self.cond_proj.bias.zero_()
        else:
            self.take_condition = False
            self.cond = nn.Parameter(
                torch.zeros(1, num_harmonics, 1),
            )

    def forward(self, f0, uv, g=None, **kwargs):
        """
        Args:
            f0: fundamental frequency shape=[N, 1, L]
            uv: unvoiced=0 / voiced=1 flag, shape=[N, 1, L]
        Output
            shape=[N, 1, L * frame_size]
        """
        dtype = f0.dtype
        f0 = f0.to(torch.float)

        # Interpolate pitch track
        f0 = F.interpolate(f0, scale_factor=self.frame_size, mode="linear")
        voiced_mask = F.interpolate(uv, scale_factor=self.frame_size, mode="linear")

        # Calculate natural number multiple frequencies
        mul = (
            (torch.arange(self.num_harmonics, device=f0.device) + 1)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        fs = f0 * mul

        # Numerical integration, generate sinusoidal harmonics
        integrated = torch.cumsum(fs / self.sample_rate, dim=2)

        if self.take_condition and g is not None:
            amps = torch.exp(self.cond_proj(g).float())
        else:
            amps = torch.exp(self.cond.float())

        rad = 2 * math.pi * (integrated % 1.0)
        noise = torch.randn(rad.shape[0], 1, rad.shape[2], device=rad.device).expand(
            rad.shape
        )

        harmonics = torch.sin(rad) * voiced_mask + noise * self.noise_scale
        harmonics = harmonics

        # switch v/uv.
        voiced_part = harmonics + noise * self.noise_scale
        unvoiced_part = noise * 0.33333

        # synthesize
        source = voiced_part * voiced_mask + unvoiced_part * (1 - voiced_mask)

        if self.normalize_amps:
            amps = F.normalize(amps, dim=1)

        source = (source * amps).sum(dim=1, keepdim=True)
        if self.post_tanh_activation:
            source = torch.tanh(source)

        source = source.to(dtype)
        return source


@derive_config
class ImpulseOscillator(nn.Module):
    def __init__(
        self, sample_rate: int = 22050, frame_size: int = 256, noise_gain: float = 0.03
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.noise_gain = noise_gain

    def forward(self, f0, uv, **kwargs):
        """
        Args:
            f0: fundamental frequency shape=[N, 1, L]
            uv: unvoiced=0 / voiced=1 flag, shape=[N, 1, L]
        Output
            shape=[N, 1, L * frame_size]
        """

        dtype = f0.dtype
        f0 = f0.to(torch.float)
        f0 = F.interpolate(f0, scale_factor=self.frame_size, mode="linear")
        voiced_mask = F.interpolate(uv, scale_factor=self.frame_size)
        rad = torch.cumsum(-f0 / self.sample_rate, dim=2)
        sawtooth = rad % 1.0
        impluse = sawtooth - sawtooth.roll(1, dims=(2))
        noise = torch.randn_like(impluse) * self.noise_gain
        source = impluse * voiced_mask + noise * (1 - voiced_mask)
        source = source.to(dtype)
        return source


# Cyclic noise oscillator from https://arxiv.org/abs/2004.02191
@derive_config
class CyclicNoiseOscillator(nn.Module):
    def __init__(
        self,
        sample_rate: int = 22050,
        frame_size: int = 256,
        base_frequency: float = 440.0,
        beta: float = 0.78,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.base_frequency = base_frequency
        self.beta = beta

        self.kernel_size = int(4.6 * self.sample_rate / self.base_frequency)
        self.pad_size = self.kernel_size - 1
        self.kernel = self.generate_kernel()

    def generate_kernel(self):
        t = torch.arange(0, self.kernel_size)[None, None, :]
        decay = torch.exp(-t * self.base_frequency / self.beta / self.sample_rate)
        noise = torch.randn_like(decay)
        kernel = noise * decay
        return kernel

    def forward(self, f0, uv, **kwargs):
        """
        Args:
            f0: fundamental frequency shape=[N, 1, L]
            uv: unvoiced=0 / voiced=1 flag, shape=[N, 1, L]
        Output
            shape=[N, 1, L * frame_size]
        """

        dtype = f0.dtype
        f0 = f0.to(torch.float)
        f0 = F.interpolate(f0, scale_factor=self.frame_size, mode="linear")
        N = f0.shape[0]
        L = f0.shape[2]
        voiced_mask = F.interpolate(uv, scale_factor=self.frame_size)
        rad = torch.cumsum(-f0 / self.sample_rate, dim=2)
        sawtooth = rad % 1.0
        impluse = sawtooth - sawtooth.roll(1, dims=(2))
        noise = torch.randn(N, 1, L, device=f0.device)
        impluse = F.pad(impluse, (self.pad_size, 0))
        cyclic_noise = F.conv1d(impluse, self.kernel)
        source = cyclic_noise * voiced_mask + (1 - voiced_mask) * noise
        source = source.to(dtype)
        return source
