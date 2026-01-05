import pytest
import torch
from torch import nn as nn

from tts_impl.functional import (
    cross_correlation,
    fft_convolve,
    framewise_fir_filter,
    impulse_train,
    sinusoidal_harmonics,
    spectral_envelope_filter,
)
from tts_impl.net.vocoder.ddsp import HomomorphicVocoder


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_frames", [100, 200])
@pytest.mark.parametrize("n_fft", [1024])
@pytest.mark.parametrize("d_periodicity", [16])
@pytest.mark.parametrize("d_spectral_envelope", [80])
@pytest.mark.parametrize("hop_length", [256])
def test_subtractive_vocoder(
    batch_size: int,
    num_frames: int,
    n_fft: int,
    d_periodicity: int,
    d_spectral_envelope: int,
    hop_length: int,
):
    sample_rate = 24000
    vocoder = HomomorphicVocoder(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    f0 = torch.ones(batch_size, num_frames) * 440.0
    per = torch.rand(batch_size, d_periodicity, num_frames)
    env = torch.rand(batch_size, d_spectral_envelope, num_frames)
    o = vocoder.forward(f0, per, env)
    assert o.shape[0] == batch_size
    assert o.shape[1] == num_frames * hop_length


def test_xcorr():
    x = torch.randn(2, 65536)
    cross_correlation(x, 1024, 256)


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
