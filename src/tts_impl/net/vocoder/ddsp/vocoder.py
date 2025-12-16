import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.functional.ddsp import estimate_minimum_phase, fft_convolve, impulse_train
from tts_impl.utils.config import derive_config


@derive_config
class HomomorphicVocoder(nn.Module):
    def __init__(
        self, hop_length: int = 256, n_fft: int | None = 1024, sample_rate: int = 24000
    ):
        super().__init__()
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        if n_fft is None:
            self.n_fft = self.hop_length * 4
        else:
            self.n_fft = n_fft
        self.register_buffer("hann_window", torch.hann_window(self.n_fft))

    def forward(
        self,
        f0: torch.Tensor,
        env_per: torch.Tensor,
        env_noi: torch.Tensor,
        rev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass to synthesize audio from acoustic features.

        Args:
            f0: Fundamental frequency [batch_size, length]
            env_per: Periodic signal spectral envelope [batch_size, fft_bin, length]
            env_noi: Noise spectral envelope [batch_size, fft_bin, length]
            rev: Optional reverb kernel [batch_size, kernel_size]

        Returns:
            Synthesized audio waveform [batch_size, length * hop_length]
        """
        dtype = env_per.dtype
        f0 = f0.float()
        env_per = env_per.float()
        env_noi = env_noi.float()
        if rev is not None:
            rev = rev.float()

        # pad
        env_per = F.pad(env_per, (1, 0))
        env_noi = F.pad(env_noi, (1, 0))

        # oscillate impulse and noise with energy normalization
        imp_scale = torch.rsqrt(
            torch.clamp_min(
                F.interpolate(
                    f0.unsqueeze(1), scale_factor=self.hop_length, mode="linear"
                ).squeeze(1),
                min=20.0,
            )
        ) * math.sqrt(self.sample_rate)
        imp = impulse_train(f0, self.hop_length, self.sample_rate) * imp_scale
        noi = torch.rand_like(imp)

        # short-time fourier transform
        imp_stft = torch.stft(
            imp,
            self.n_fft,
            hop_length=self.hop_length,
            window=self.hann_window,
            return_complex=True,
        )
        noi_stft = torch.stft(
            noi,
            self.n_fft,
            hop_length=self.hop_length,
            window=self.hann_window,
            return_complex=True,
        )

        # excitation and filter
        env_per *= (F.pad(f0[:, None, :], (1, 0)) > 20.0).to(
            torch.float
        )  # set periodicity=0 if unvoiced.
        voi_stft = imp_stft * estimate_minimum_phase(env_per) + noi_stft * env_noi

        # inverse STFT
        voi = torch.istft(
            voi_stft,
            self.n_fft,
            self.hop_length,
            window=self.hann_window,
        )

        # apply post filter (reverb)
        if rev is not None:
            voi = fft_convolve(voi, rev)
        return voi.to(dtype)
