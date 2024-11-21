from typing import Optional

import torch
import torch.nn as nn
import torchaudio


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        eps=1e-5,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate,
            n_fft,
            n_fft,
            hop_length,
            fmin,
            fmax,
            n_mels=n_mels,
            window_fn=torch.hann_window,
        )
        self.eps = eps

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        x = self.mel_spec(signal)
        x = self.safe_log(x)
        return x

    def safe_log(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log(x + self.eps)
