import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tts_impl.functional.ddsp import estimate_minimum_phase, fft_convolve, impulse_train
from tts_impl.utils.config import derive_config
from torchaudio.transforms import MelScale, InverseMelScale


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

    def synthesize(
        self,
        f0: Tensor,
        periodicity: Tensor,
        vocal_tract: Tensor,
        reverb: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            f0: shape=(batch_size, n_frames)
            periodicity: shape=(batch_size, fft_bin, n_frames)
            vocal_tract: shape=(batch_size, fft_bin, n_frames)
            reverb: shape=(batch_size, filter_size), Optional, post-filter

        Returns:
            output: synthesized wave
        """

        # cast to 32-bit float for stability.
        dtype = f0.dtype
        f0 = f0.to(torch.float)
        periodicity = periodicity.to(torch.float)
        vocal_tract = vocal_tract.to(torch.float)

        # pad
        vocal_tract = F.pad(vocal_tract, (1, 0))
        periodicity = F.pad(periodicity, (1, 0))

        # oscillate impulse and noise
        with torch.no_grad():
            imp_scale = torch.rsqrt(
                torch.clamp_min(
                    F.interpolate(
                        f0.unsqueeze(1), scale_factor=self.hop_length, mode="linear"
                    ).squeeze(1),
                    min=20.0,
                )
            )
            noi_scale = 1.0 / math.sqrt(self.sample_rate)
            imp = impulse_train(f0, self.hop_length, self.sample_rate) * imp_scale
            noi = (torch.rand_like(imp) * 2 - 1) * noi_scale

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

        # excitation signal
        excitation_stft = imp_stft * periodicity + noi_stft * (1 - periodicity)

        # estimate minimum(causal) phase. (optional)
        if self.min_phase:
            vocal_tract = estimate_minimum_phase(vocal_tract)

        # apply the filter to impulse / noise, and add them.
        voi_stft = excitation_stft * vocal_tract

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
