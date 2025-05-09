import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchaudio.transforms import InverseMelScale
from tts_impl.functional.ddsp import estimate_minimum_phase, fft_convolve, impulse_train
from tts_impl.utils.config import derive_config


@derive_config
class SubtractiveVocoder(nn.Module):
    """
    Subtractive DDSP Vocoder that likely purposed at Meta's [paper](https://arxiv.org/abs/2401.10460)
    """

    def __init__(
        self,
        dim_periodicity: int = 12,
        n_mels: int = 80,
        sample_rate: int = 24000,
        hop_length: int = 256,
        n_fft: int = 1024,
        min_phase: bool = True,
    ):
        """
        Args:
            dim_periodicity: dimension of periodicity, int
            n_mels: mel-bandwidth of vocal tract filter, int
            sample_rate: int
            hop_length: hop length of STFT, int
            n_fft: fft window size of STFT, int
            min_phase: flag to use minimum phase, bool, default=True
        """
        super().__init__()
        self.dim_periodicity = dim_periodicity
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.fft_bin = n_fft // 2 + 1
        self.hop_length = hop_length
        self.min_phase = min_phase

        self.per2spec = InverseMelScale(self.fft_bin, dim_periodicity, sample_rate)
        self.env2spec = InverseMelScale(self.fft_bin, n_mels, sample_rate)
        self.hann_window = nn.Parameter(torch.hann_window(n_fft))

    def forward(
        self,
        f0: Tensor,
        periodicity: Tensor,
        vocal_tract: Tensor,
        vocal_cord: Optional[Tensor] = None,
        reverb: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            f0: shape=(batch_size, n_frames)
            periodicity: shape=(batch_size, dim_periodicity, n_frames), periodicity
            vocal_tract: shape=(batch_size, fft_bin, n_frames), spectral envelope of vocal tract
            vocal_cord: shape=(batch_size, kernel_size), Optional, vocal cord impulse response
            reverb: shape=(batch_size, filter_size), Optional, room reverb impulse response

        Returns:
            output: synthesized wave
        """

        # cast to 32-bit float for stability.
        dtype = periodicity.dtype
        periodicity = periodicity.to(torch.float)
        vocal_tract = vocal_tract.to(torch.float)

        # oscillate and calculate complex spectra.
        with torch.no_grad():
            # oscillate impulse train and noise
            imp_scale = torch.rsqrt(
                F.interpolate(f0.unsqueeze(1), scale_factor=self.hop_length).squeeze(1)
            ) * math.sqrt(self.sample_rate)
            imp = impulse_train(f0, self.hop_length, self.sample_rate) * imp_scale
            noi = torch.rand_like(imp)

        # vocal cord filter
        if vocal_cord is not None:
            imp = F.pad(imp[None, :, :], (vocal_cord.shape[1] - 1, 0))
            imp = F.conv1d(
                imp, vocal_cord[:, None, :], groups=vocal_cord.shape[0]
            ).squeeze(0)

        # short-time fourier transform
        imp_stft = torch.stft(
            imp, self.n_fft, window=self.hann_window, return_complex=True
        )
        noi_stft = torch.stft(
            noi, self.n_fft, window=self.hann_window, return_complex=True
        )

        # replace impulse to noise if unvoiced.
        imp_stft += noi_stft * (F.pad(f0[:, None, :], (1, 0)) < 20.0).to(torch.float)

        # Convert mel-spectral envelope to linear-spectral envelope.
        vocal_tract_linear = self.env2spec(vocal_tract)

        # Merge periodicity / apeoridicity.
        kernel_imp = self.per2spec(periodicity) * vocal_tract_linear
        kernel_noi = self.per2spec(1 - periodicity) * vocal_tract_linear

        # estimate minimum(causal) phase. (optional)
        if self.min_phase:
            kernel_imp = estimate_minimum_phase(kernel_imp)
            kernel_noi = estimate_minimum_phase(kernel_noi)

        # pad
        kernel_imp = F.pad(kernel_imp, (1, 0))
        kernel_noi = F.pad(kernel_noi, (1, 0))

        # apply the filter to impulse / noise, and add them.
        voi_stft = imp_stft * kernel_imp + noi_stft * kernel_noi

        # inverse STFT
        voi = torch.istft(
            voi_stft, self.n_fft, self.hop_length, window=self.hann_window
        )

        # apply post filter. (optional)
        if reverb is not None:
            voi = fft_convolve(voi, reverb)

        # cast back to the original dtype.
        voi = voi.to(dtype)
        return voi
