from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def framewise_fir_filter(
    signal: torch.Tensor, filter: torch.Tensor, n_fft: int, hop_length: int, center:bool=True
) -> torch.Tensor:
    """
    args:
        signal: [batch_size, length * hop_length]
        filter: [batch_size, n_fft, length]
    outputs:
        signal: [batch_size, length * hop_length]
    """

    dtype = signal.dtype

    x = signal.to(torch.float)
    window = torch.hann_window(n_fft, device=x.device)
    x_stft = torch.stft(x, n_fft, hop_length, n_fft, window, center, return_complex=True)
    filter_stft = torch.fft.rfft(filter, dim=1)
    x_stft = x_stft * filter_stft
    x = torch.istft(x_stft, n_fft, hop_length, n_fft, window, center, return_complex=True)
    signal = x.to(dtype)
    return signal


def spectral_envelope_filter(
    signal: torch.Tensor, envelope: torch.Tensor, n_fft: int, hop_length: int
) -> torch.Tensor:
    """
    args:
        signal: [batch_size, length * hop_length]
        envelope: [batch_size, n_fft, length]
    outputs:
        signal: [batch_size, length * hop_length]
    """
    raise "UNIMPLEMENTED"


def impulse_train(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: float,
    uv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    args:
        f0: [batch_size, length]
        uv: [batch_size, length]
    outputs:
        signal: [batch_size, 1, length * hop_length]
    """
    f0 = f0.unsqueeze(1)
    f0 = F.interpolate(f0, scale_factor=hop_length, mode="linear")
    if uv is not None:
        uv = uv.to(f0.dtype)
        uv = uv.unsqueeze(1)
        uv = F.interpolate(uv, scale_factor=hop_length)
        f0 = f0 * uv
    I = torch.cumsum(f0, dim=2)
    sawtooth = (I / sample_rate) % 1.0
    impulse = sawtooth - sawtooth.roll(-1, dims=(2)) + (f0 / sample_rate)
    return impulse