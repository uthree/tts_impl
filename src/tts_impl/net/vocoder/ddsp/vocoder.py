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
    Subtractive DDSP Vocoder
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = 256,
        n_fft: int = 1024,
        min_phase: bool = True,
        dim_periodicity: int = 16,
        dim_envelope: int = 80,
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
        self.dim_periodicity = dim_periodicity
        self.dim_envelope = dim_envelope

        self.register_buffer("hann_window", torch.hann_window(n_fft))
        self.per2spec = InverseMelScale(
            n_stft=self.fft_bin, n_mels=dim_periodicity, sample_rate=sample_rate
        )
        self.env2spec = InverseMelScale(
            n_stft=self.fft_bin, n_mels=dim_envelope, sample_rate=sample_rate
        )

    def synthesize(
        self,
        f0: Tensor,
        periodicity: Tensor,
        envelope: Tensor,
        reverb: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            f0: shape=(batch_size, n_frames)
            periodicity: shape=(batch_size, fft_bin, n_frames)
            envelope: shape=(batch_size, fft_bin, n_frames)
            reverb: shape=(batch_size, filter_size), Optional, post-filter

        Returns:
            output: synthesized wave
        """

        # cast to 32-bit float for stability.
        dtype = f0.dtype
        f0 = f0.to(torch.float)
        periodicity = periodicity.to(torch.float)
        envelope = envelope.to(torch.float)

        # pad
        envelope = F.pad(envelope, (1, 0))
        periodicity = F.pad(periodicity, (1, 0))

        # to linear scale
        kernel_imp = self.env2spec(envelope) * self.per2spec(periodicity)
        kernel_noi = self.env2spec(envelope) * self.per2spec(1 - periodicity)

        # estimate minimum(causal) phase. (optional)
        if self.min_phase:
            kernel_imp = estimate_minimum_phase(kernel_imp)

        # oscillate impulse and noise
        with torch.no_grad():
            imp_scale = torch.rsqrt(
                torch.clamp_min(
                    F.interpolate(
                        f0.unsqueeze(1), scale_factor=self.hop_length, mode="linear"
                    ).squeeze(1),
                    min=20.0,
                )
            ) * math.sqrt(self.sample_rate)
            noi_scale = 2.0
            imp = impulse_train(f0, self.hop_length, self.sample_rate) * imp_scale
            noi = torch.rand_like(imp) * noi_scale

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

        # apply the filter to impulse / noise, and add them.
        voi_stft = noi_stft * kernel_noi + imp_stft * kernel_imp

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
