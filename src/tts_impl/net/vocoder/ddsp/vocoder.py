import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tts_impl.functional.ddsp import estimate_minimum_phase, fft_convolve, impulse_train
from tts_impl.utils.config import derive_config
import torchaudio


@derive_config
class SubtractiveVocoder(nn.Module):
    """
    Subtractive DDSP Vocoder
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = 256,
        dim_periodicity: int = 16,
        n_fft: int = 1024,
    ):
        """
        Args:
            sample_rate: int
            hop_length: hop length of STFT, int
            n_fft: fft window size of STFT, int
            dim_periodicity: periodicity dim, int
            min_phase: flag to use minimum phase, bool, default=True
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.fft_bin = n_fft // 2 + 1
        self.hop_length = hop_length
        self.dim_periodicity = dim_periodicity
        self.register_buffer("hann_window", torch.hann_window(n_fft))
        self.per2bins = torchaudio.transforms.InverseMelScale(self.fft_bin, dim_periodicity, sample_rate)

    def synthesize(
        self,
        f0: Tensor,
        per: Tensor,
        env: Tensor,
        reverb: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            f0: shape=(batch_size, n_frames)
            per: shape=(batch_size, fft_bin, n_frames)
            env: shape=(batch_size, fft_bin, n_frames)
            reverb: shape=(batch_size, filter_size), Optional, post-filter

        Returns:
            output: synthesized wave
        """

        # cast to 32-bit float for stability.
        dtype = f0.dtype
        f0 = f0.to(torch.float)
        per = per.float()
        env = env.float()

        # pad
        per = F.pad(per, (1, 0))
        env = F.pad(env, (1, 0))

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
        per *= (F.pad(f0[:, None, :], (1, 0)) > 20.0).to(torch.float) # set periodicity=0 if unvoiced.
        periodicity = self.per2bins(per)
        aperiodicity = self.per2bins(1-per)
        excitation = aperiodicity * noi_stft + periodicity * imp_stft
        voi_stft = excitation * estimate_minimum_phase(env)

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
