import torch
import torch.nn as nn
import torch.nn.functional as F

from tts_impl.functional import framewise_fir_filter, fft_convolve, sinusoidal_harmonics, impulse_train, spectral_envelope_filter


def test_framewise_fir_filter():
    x = torch.randn(2, 65536)
    h = torch.randn(2, 1024, 256)
    o = framewise_fir_filter(x, h, 1024, 256)
    assert o.shape == x.shape


def test_harmonics():
    harmonics = sinusoidal_harmonics(torch.exp(torch.randn(2, 1000)), 8, 22050, 256)