from typing import Optional

import torch
import torchaudio
from torch import nn as nn
from torch.nn import functional as F
from tts_impl.utils.config import derive_config


@derive_config
class LogMelSpectrogram(nn.Module):
    """
    A module to calculate the logarithmic mel spectrogram
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float | None = None,
        eps: float = 1e-8,
    ):
        """
        Args:
            sample_rate: int, sample rate of target waveform.
            n_fft: FFT window size.
            hop_length: FFT hop length.
            n_mels: number of mel bands.
            fmin: minimum frequency.
            fmax: maximum frequency.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.max = fmax
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate,
            n_fft,
            n_fft,
            hop_length,
            fmin,
            fmax,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            center=False,
            normalized=False,
        )
        self.eps = eps

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute the mel spectrogram.
        Given a batched signal it returns the batched spectrum, otherwise it returns the unbatched spectrum.

        Args:
            signal: Tensor, shape=(batch_size, length)

        Returns:
            spectrogram: Tensor, shape=(batch_size, n_mels, length // hop_length)
        """
        x = signal.float()
        x = F.pad(
            x,
            (
                int((self.n_fft - self.hop_length) / 2),
                int((self.n_fft - self.hop_length) / 2),
            ),
            mode="reflect",
        )
        x = self.mel_spec(x)
        x = self.safe_log(x)
        return x

    def safe_log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))
