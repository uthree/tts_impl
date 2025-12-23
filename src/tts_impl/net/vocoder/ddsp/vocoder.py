import math

import torch
from torch import nn as nn
from torch.fft import fft
from torch.nn import functional as F
from torchaudio.transforms import InverseMelScale

from tts_impl.functional.ddsp import estimate_minimum_phase, fft_convolve, impulse_train
from tts_impl.utils.config import derive_config


@derive_config
class HomomorphicVocoder(nn.Module):
    def __init__(
        self,
        hop_length: int = 256,
        n_fft: int | None = 1024,
        sample_rate: int = 24000,
        d_spectral_envelope: int = 80,
        d_periodicity: int = 16,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.d_spectral_envelope = d_spectral_envelope
        self.d_periodicity = d_periodicity

        if n_fft is None:
            self.n_fft = self.hop_length * 4
        else:
            self.n_fft = n_fft

        fft_bins = self.n_fft // 2 + 1
        self.expand_periodicity = InverseMelScale(
            n_stft=fft_bins, sample_rate=self.sample_rate, n_mels=self.d_periodicity
        )
        self.expand_spectral_envelope = InverseMelScale(
            n_stft=fft_bins,
            sample_rate=self.sample_rate,
            n_mels=self.d_spectral_envelope,
        )
        self.register_buffer("hann_window", torch.hann_window(self.n_fft))

    def forward(
        self,
        f0: torch.Tensor,
        periodicity: torch.Tensor,
        spectral_envelope: torch.Tensor,
        rev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass to synthesize audio from acoustic features.

        Args:
            f0: Fundamental frequency [batch_size, length]
            periodicity: band periodicity [batch_size, d_periodicity, length]
            spectral_envelope: Noise spectral envelope [batch_size, d_spectral_envelope, length]
            rev: Optional reverb kernel [batch_size, kernel_size]

        Returns:
            Synthesized audio waveform [batch_size, length * hop_length]
        """

        dtype = periodicity.dtype
        f0 = f0.float()
        periodicity = periodicity.float()
        spectral_envelope = spectral_envelope.float()
        if rev is not None:
            rev = rev.float()

        # pad
        periodicity = F.pad(periodicity, (1, 0))
        spectral_envelope = F.pad(spectral_envelope, (1, 0))

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
        imp = imp.detach()
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

        # expand band periodicity and spectral envelope
        per = self.expand_periodicity(periodicity)
        env = self.expand_spectral_envelope(spectral_envelope)

        per = per * (F.pad(f0[:, None, :], (1, 0)) > 20.0).to(
            torch.float
        )  # set periodicity=0 if unvoiced.

        voi_stft = (
            imp_stft * estimate_minimum_phase(per * env) + noi_stft * (1 - per) * env
        )

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
