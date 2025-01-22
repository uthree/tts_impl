# https://arxiv.org/abs/2401.10460
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks

from tts_impl.functional.ddsp import impulse_train


class UltraLighweightDdsp(nn.Module):
    """
    Unofficial implementation of Meta AI's Ultra-Lightweight DDSP Vocoder
    based https://arxiv.org/abs/2401.10460
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = 128,
        n_fft: int = 512,
        n_mels: int = 12,
        fmin: float = 0,
        fmax: float = 8000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fft_bin = n_fft // 2 + 1
        self.fmin = fmin
        self.fmax = fmax

        mel_fbank = melscale_fbanks(
            self.fft_bin, fmin, fmax, n_mels, self.sample_rate
        )  # [n_freqs, n_mels]
        self.register_buffer(
            "mel_fbank", mel_fbank
        )  # register as non-trainable parameter

    def forward(
        self, f0: torch.Tensor, p: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f0: shape=(batch_size, L)
            p: shape=(batch_size, n_mels, L)
            v: shape=(fft_bin, n_mels, L)

            where, fft_bin = n_fft // 2 + 1.

        Returns:
            x_hat: shape=(batch_size, L*hop_length)
        """

        # Osscilate impulse
        e_imp = impulse_train(
            f0, self.hop_length, self.sample_rate
        )  # [B, L * hop_length]
        # Energy normaliziaton
        m = torch.rsqrt(
            F.interpolate(f0.unsqueeze(1), scale_factor=self.hop_length).squeeze(1)
        )
        e_imp = e_imp * m

        # Oscillate noise
        e_noise = torch.rand_like(e_imp) * math.isqrt(
            self.sample_rate
        )  # [B, L * hop_length]

        # STFT
        e_imp = torch.stft(
            e_imp, self.n_fft, self.hop_length, return_complex=True
        )  # Complex, [B, C, L + 1]
        e_noise = torch.stft(
            e_noise, self.n_fft, self.hop_length, return_complex=True
        )  # Complex, [B, C, L + 1]

        # expand p
        p = torch.matmul(self.mel_fbank, p)
        p = F.pad(p, (1, 0), mode="replicate")

        # source signal
        s = p * e_imp + (1 - p) * e_noise

        # vocal tract filter
        v = F.pad(v, (1, 0), mode="replicate")
        x_hat_stft = s * v

        # iSTFT
        x_hat = torch.istft(x_hat_stft, self.n_fft, self.hop_length)
        return x_hat
