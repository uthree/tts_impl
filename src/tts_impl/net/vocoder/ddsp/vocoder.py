import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tts_impl.functional.ddsp import estimate_minimum_phase, fft_convolve, impulse_train
from tts_impl.utils.config import derive_config


@derive_config
class SubtractiveVocoder(nn.Module):
    """
    Subtractive DDSP Vocoder
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = 256,
        n_fft: int = 1024,
        min_phase: bool = True,
    ):
        """
        Args:
            sample_rate: int
            hop_length: hop length of STFT, int
            n_fft: fft window size of STFT, int
            min_phase: flag to use minimum phase, bool, default=True
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.fft_bin = n_fft // 2 + 1
        self.hop_length = hop_length
        self.min_phase = min_phase

        self.register_buffer("hann_window", torch.hann_window(n_fft))

    def forward(
        self,
        f0: Tensor,
        envelope_imp: Tensor,
        envelope_noi: Tensor,
        reverb: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            f0: shape=(batch_size, n_frames)
            envelope_imp: shape=(batch_size, fft_bin, n_frames), spectral envelope for impulse
            envelope_noi: shape=(batch_size, fft_bin, n_frames), spectral envelope for noise
            vocal_cord: shape=(batch_size, kernel_size), Optional, vocal cord impulse response
            reverb: shape=(batch_size, filter_size), Optional, room reverb impulse response

        Returns:
            output: synthesized wave
        """

        # cast to 32-bit float for stability.
        dtype = f0.dtype
        kernel_noi = envelope_noi.to(torch.float)
        kernel_imp = envelope_imp.to(torch.float)

        # oscillate and calculate complex spectra.
        with torch.no_grad():
            # oscillate impulse train and noise
            imp_scale = torch.rsqrt(
                torch.clamp_min(
                    F.interpolate(
                        f0.unsqueeze(1), scale_factor=self.hop_length, mode="linear"
                    ).squeeze(1),
                    min=20.0,
                )
            ) * math.sqrt(self.sample_rate)
            imp = impulse_train(f0, self.hop_length, self.sample_rate) * imp_scale
            noi = torch.randn_like(imp) * 0.33333

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

        # replace impulse to noise if unvoiced.
        imp_stft += noi_stft * (F.pad(f0[:, None, :], (1, 0)) < 20.0).to(torch.float)

        # estimate minimum(causal) phase. (optional)
        if self.min_phase:
            kernel_imp = estimate_minimum_phase(kernel_imp)

        # pad
        kernel_imp = F.pad(kernel_imp, (1, 0))
        kernel_noi = F.pad(kernel_noi, (1, 0))

        # apply the filter to impulse / noise, and add them.
        voi_stft = imp_stft * kernel_imp + noi_stft * kernel_noi

        # inverse STFT
        voi = torch.istft(
            voi_stft,
            self.n_fft,
            self.hop_length,
            window=self.hann_window,
        )

        # apply post filter. (optional)
        if reverb is not None:
            voi = fft_convolve(voi, reverb)

        # cast back to the original dtype.
        voi = voi.to(dtype)
        return voi
