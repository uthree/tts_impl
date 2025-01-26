import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple
from torch import Tensor


class STFT(nn.Module):
    """
    An implementation of short time fourier transform using `torch.nn.functional.conv1d` for onnx compatibility.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        window: Literal["rectangle", "hann"] = "hann",
        padding: Literal["none", "causal", "center"] = "None",
    ):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.padding = padding
        if window == "rectangle":
            w = None
        elif window == "hann":
            w = torch.hann_window(n_fft)
        else:
            raise RuntimeError("Invalid window")
        self.register_buffer("kernel", self.stft_kernel(n_fft, window=w))

    def stft_kernel(self, n_fft: int, window=None) -> Tensor:
        fft_bin = n_fft // 2 + 1
        xt = (
            torch.linspace(0, 1.0, n_fft)[None, None, :]
            * torch.arange(fft_bin)[:, None, None]
            * 2
            * torch.pi
        )
        kernel = torch.cat([torch.cos(xt), torch.sin(xt)], dim=0)
        if window is not None:
            kernel = kernel * window[None, None, :]
        return kernel

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: shape=(batch_size, 1, Length)

        Returns:
            real: shape=(batch_size, fft_bin, num_frames)
            imag: shape=(batch_size, fft_bin, num_frames)
        """
        if self.padding == "center":
            left_pad = self.n_fft // 2
            right_pad = self.n_fft - left_pad
            x = F.pad(x, (left_pad, right_pad))
        elif self.padding == "causal":
            x = F.pad(x, (0, self.n_fft - self.hop_length))
        x = F.conv1d(x, self.kernel, stride=self.hop_length)
        real, imag = torch.chunk(x, 2, dim=1)
        return real, imag


class ISTFT(nn.Module):
    """
    An implementation of inverse short time fourier transform using `torch.nn.functional.conv_transpose1d` for onnx compatibility.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        window: Literal["rectangle", "hann"] = "hann",
        padding: Literal["none", "causal", "center"] = "None",
    ):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.padding = padding
        if window == "rectangle":
            w = None
        elif window == "hann":
            w = torch.hann_window(n_fft)
        else:
            raise RuntimeError("Invalid window")
        self.register_buffer("kernel", self.istft_kernel(n_fft, window=w))

    def istft_kernel(self, n_fft: int, window=None):
        fft_bin = n_fft // 2 + 1
        xt = (
            torch.linspace(0, 1.0, n_fft)[None, None, :]
            * torch.arange(fft_bin)[:, None, None]
            * 2
            * torch.pi
        )
        kernel = torch.cat([torch.cos(xt), torch.sin(xt)], dim=0)
        if window is not None:
            kernel = kernel * window[None, None, :] / window.sum()
        else:
            kernel /= n_fft
        return kernel

    def forward(self, real: Tensor, imag: Tensor) -> Tensor:
        """
        Args:
            real: shape=(batch_size, fft_bin, num_frames)
            imag: shape=(batch_size, fft_bin, num_frames)
        Returns:
            x: shape=(batch_size, 1, Length)
        """
        x = torch.cat([real, imag], dim=1)
        x = F.conv_transpose1d(x, self.kernel, stride=self.hop_length)
        if self.padding == "center":
            left_pad = self.n_fft // 2
            right_pad = self.n_fft - left_pad
            x = x[:, :, left_pad:-right_pad]
        elif self.padding == "causal":
            x = x[:, :, : -self.n_fft + self.hop_length]
        return x
