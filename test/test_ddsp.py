import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.functional import (
    fft_convolve,
    framewise_fir_filter,
    impulse_train,
    sinusoidal_harmonics,
    spectral_envelope_filter,
)


def test_framewise_fir_filter():
    x = torch.randn(2, 65536)
    h = torch.randn(2, 1024, 256)
    o = framewise_fir_filter(x, h, 1024, 256)
    assert o.shape == x.shape


def test_harmonics():
    harmonics = sinusoidal_harmonics(torch.exp(torch.randn(2, 1000)), 8, 22050, 256)


def test_fft_convolve():
    a = torch.randn(1, 2, 10000)
    b = torch.randn(1, 2, 100)
    c = fft_convolve(a, b)
    assert a.shape == c.shape


def test_impulse_train():
    f0 = torch.randn(1, 100)
    s = impulse_train(f0, 480, 48000)


def test_spectral_envelope_filter():
    senv = torch.randn(1, 513, 256)
    noi = torch.randn(1, 65536)
    filtered = spectral_envelope_filter(noi, senv, 1024, 256)
