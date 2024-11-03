from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def framewise_fir_filter(
    input_signal: torch.Tensor, filter: torch.Tensor, n_fft: int, hop_length: int
) -> torch.Tensor:
    raise "UNIMPLEMENTED!!!!"


def spectral_envelope_filter(
    input_signal: torch.Tensor, envelope: torch.Tensor, n_fft: int, hop_length: int
) -> torch.Tensor:
    raise "UNIMPLEMENTED!!!!"


def impulse_train(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: float,
    uv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    f0 = F.interpolate(f0, scale_factor=hop_length, mode="linear")
    if uv is not None:
        uv = uv.to(f0.dtype)
        uv = F.interpolate(uv, scale_factor=hop_length)
        f0 = f0 * uv
    I = torch.cumsum(f0, dim=2)
    sawtooth = (I / sample_rate) % 1.0
    impulse = sawtooth - sawtooth.roll(-1, dims=(2)) + (f0 / sample_rate)
    return impulse
