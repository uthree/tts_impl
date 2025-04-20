from typing import Optional

import torch
import torch.nn.functional as F


def framewise_fir_filter(
    signal: torch.Tensor,
    filter: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """
    apply different FIR filter frame-wise.

    args:
        signal: [batch_size, length * hop_length]
        filter: [batch_size, n_fft, length]
        n_fft: int
        hop_length: int

    outputs:
        signal: [batch_size, length * hop_length]
    """

    dtype = signal.dtype
    x = signal.to(torch.float)
    window = torch.hann_window(n_fft, device=x.device)
    x_stft = torch.stft(
        x, n_fft, hop_length, n_fft, window, center=True, return_complex=True
    )
    filter = F.pad(filter, (0, 1), mode="replicate")
    filter_stft = torch.fft.rfft(filter, dim=1)
    x_stft = x_stft * filter_stft
    x = torch.istft(x_stft, n_fft, hop_length, n_fft, window, center=True)
    signal = x.to(dtype)
    return signal


def spectral_envelope_filter(
    signal: torch.Tensor, envelope: torch.Tensor, n_fft: int, hop_length: int
) -> torch.Tensor:
    """
    args:
        signal: [batch_size, length * hop_length]
        envelope: [batch_size, fft_bin, length], where fft_bin = n_fft // 2 + 1
    outputs:
        signal: [batch_size, length * hop_length]
    """
    dtype = signal.dtype
    signal = signal.to(torch.float)
    envelope = envelope.to(torch.float)
    window = torch.hann_window(n_fft, device=signal.device)
    signal_stft = (
        torch.stft(signal, n_fft, hop_length, window=window, return_complex=True)[
            :, :, 1:
        ]
        * envelope
    )
    signal_stft = F.pad(signal_stft, (0, 1))
    signal = torch.istft(signal_stft, n_fft, hop_length, window=window)
    signal = signal.to(dtype)
    return signal


def impulse_train(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    uv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    args:
        f0: [batch_size, length]
        sample_rate: int
        hop_length: int
        uv: [batch_size, length]
    outputs:
        signal: [batch_size, length * hop_length]
    """
    with torch.no_grad():
        dtype = f0.dtype
        f0 = f0.to(torch.float)
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
        impulse = impulse.squeeze(1)
    impulse = impulse.to(dtype)
    impulse = impulse.detach()
    return impulse


def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """ "
    depthwise causal convolution using fft for performance

    args:
        signal: [..., length]
        kernel: [..., kernel_size]
    outputs:
        signal: [..., length]
    """
    dtype = signal.dtype
    signal = signal.to(torch.float)
    kernel = kernel.to(torch.float)

    kernel = F.pad(kernel, (0, signal.shape[-1] - kernel.shape[-1]))

    signal = F.pad(signal, (0, signal.shape[-1]))
    kernel = F.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2 :]

    output = output.to(dtype)
    return output


def sinusoidal_harmonics(
    f0: torch.Tensor,
    num_harmonics: int,
    sample_rate: int,
    hop_length: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> torch.Tensor:
    """
    generate sinusoidal harmonic signal
    args:
        f0: [batch_size, length]
        num_harmonics: int
        hop_length: int
        sample_rate: int
        fmin: float
        fmax: float
    outputs:
        harmonics [batch_size, num_harmonics, length * hop_length]
    """
    with torch.no_grad():
        device = f0.device
        dtype = f0.dtype
        f0 = f0.to(torch.float)
        f0 = F.relu(f0)
        f0 = f0.unsqueeze(1)
        mul = (torch.arange(num_harmonics, device=device) + 1).unsqueeze(0).unsqueeze(2)
        fs = F.interpolate(f0, scale_factor=hop_length, mode="linear") * mul
        uv = (f0 > fmin).to(torch.float)
        if fmax is not None:
            uv = uv * (f0 < fmax).to(torch.float)
        uv = F.interpolate(uv, scale_factor=hop_length, mode="linear")
        I = torch.cumsum(fs / sample_rate, dim=2)  # integration
        theta = 2 * torch.pi * (I % 1)  # phase
        harmonics = (torch.sin(theta) * uv).sum(dim=1, keepdim=True)
        harmonics = harmonics.to(dtype)
    harmonics = harmonics.detach()
    return harmonics


def cross_correlation(
    signal: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """
    calculate cross correlation

    Args:
        signal: [batch_size, length * hop_length]
        n_fft: int
        hop_length: int

    Returns:
        x_corr: [batch_size, fft_bin, length]
    """

    dtype = signal.dtype
    x = signal.to(torch.float)
    window = torch.hann_window(n_fft, device=x.device)
    x_stft = torch.stft(x, n_fft, hop_length, n_fft, window, return_complex=True)[
        :, :, 1:
    ]
    x_corr = torch.fft.irfft(x_stft * x_stft.conj(), dim=1).to(dtype)
    return x_corr
