import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torchaudio.transforms import InverseMelScale
from tts_impl.functional.ddsp import (
    impulse_train,
    fft_convolve,
)
import math


class DdspVocoder(nn.Module):
    def __init__(
        self,
        dim_periodicity: int = 12,
        sample_rate: int = 24000,
        hop_length: int = 256,
        n_fft: int = 1024,
        min_phase: bool = False,
    ):
        super().__init__()
        self.dim_periodicity = dim_periodicity
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.fft_bin = n_fft // 2 + 1
        self.hop_length = hop_length
        self.min_phase = min_phase

        self.per2spec = InverseMelScale(self.fft_bin, dim_periodicity, sample_rate)
        self.hann_window = nn.Parameter(torch.hann_window(n_fft))

    def forward(
        self,
        f0: Tensor,
        periodicity: Tensor,
        spectral_envelope: Tensor,
        post_filter: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            f0: shape=(batch_size, n_frames)
            periodicity: shape=(batch_size, dim_periodicity, n_frames)
            spectral_envelope: shape=(batch_size, fft_bin, n_frames)
            post_filter: shape=(batch_size, filter_size)

        Returns:
            output: synthesized wave
        """
        dtype = periodicity.dtype
        periodicity = periodicity.to(torch.float)
        spectral_envelope = spectral_envelope.to(torch.float)
        post_filter = post_filter.to(torch.float)

        with torch.no_grad():
            # oscillate impulse train and gaussian noise
            imp = impulse_train(f0, self.hop_length, self.sample_rate) * F.interpolate(torch.rsqrt(torch.clamp_min(f0, 20.0)[:, None, :]), scale_factor=self.hop_length).squeeze(1)
            noi = (torch.rand_like(imp) - 0.5) * 2 / math.sqrt(self.sample_rate)

            # fourier transform
            imp_stft = torch.stft(
                imp, self.n_fft, window=self.hann_window, return_complex=True
            )
            noi_stft = torch.stft(
                noi, self.n_fft, window=self.hann_window, return_complex=True
            )

        # excitation
        imp_stft = imp_stft * F.pad(self.per2spec(periodicity), (1, 0))
        noi_stft = noi_stft * F.pad(self.per2spec(1.0-periodicity), (1, 0))
        exc_stft = imp_stft + noi_stft

        # FIR Filter
        if self.min_phase:
            cepst = torch.fft.irfft(torch.log(torch.clamp_min(spectral_envelope, 1e-8)), dim=1)
            h = self.n_fft // 2
            cepst[:, h:-1] *= 2.0
            cepst[:, 1:h] *= 0.0
            spectral_envelope = torch.exp(torch.fft.rfft(cepst, dim=1))
        voi_stft = exc_stft * F.pad(spectral_envelope, (1, 0))

        # inverse STFT
        voi = torch.istft(voi_stft, self.n_fft, self.hop_length, window=self.hann_window)

        # apply post filter
        if post_filter is not None:
            voi = fft_convolve(voi, post_filter)
        
        voi = voi.to(dtype)
        return voi