import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.utils.config import derive_config


# HnNSF Module from https://arxiv.org/pdf/1904.12088
@derive_config
class HarmonicNoiseOscillator(nn.Module):
    def __init__(
        self,
        sample_rate: int = 22050,
        frame_size: int = 256,
        num_harmonics=8,
        noise_scale=0.1,
        gin_channels: int = 0,
        normalize_amps: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.num_harmonics = num_harmonics
        self.noise_scale = noise_scale
        self.normalize_amps = normalize_amps

        if gin_channels > 0:
            self.take_condition = True
            self.cond = nn.Conv1d(gin_channels, num_harmonics, 1)
        else:
            self.take_condition = False
            self.weight = nn.Parameter(torch.zeros(1, num_harmonics, 1))

    def forward(self, f0, uv, g=None, **kwargs):
        """
        Args:
            f0: fundamental frequency shape=[N, 1, L]
            uv: unvoiced=0 / voiced=1 flag, shape=[N, 1, L]
        Output
            shape=[N, 1, L * frame_size]
        """

        with torch.no_grad():
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
            fs_mask = fs < (self.sample_rate / 4.0)
            integrated = torch.cumsum(fs / self.sample_rate, dim=2)
            rad = 2 * math.pi * ((integrated) % 1)
            noise = torch.randn(
                rad.shape[0], 1, rad.shape[2], device=rad.device
            ).expand(rad.shape)
            harmonics = (
                torch.sin(rad) * fs_mask * voiced_mask + noise * self.noise_scale
            )

            # switch v/uv.
            voiced_part = harmonics + noise * self.noise_scale
            unvoiced_part = noise * 0.333
            source = voiced_part * voiced_mask + unvoiced_part * (1 - voiced_mask)

        if self.take_condition and g is not None:
            w = torch.exp(self.cond(g))
        else:
            w = torch.exp(self.weight)

        if self.normalize_amps:
            w = F.normalize(w, dim=1)

        source = (source * w).sum(dim=1, keepdim=True)
        source = torch.tanh(source)
        return source


@derive_config
class ImpulseOscillator(nn.Module):
    def __init__(
        self,
        sample_rate: int = 22050,
        frame_size: int = 256,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size

    def forward(self, f0, uv, **kwargs):
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
