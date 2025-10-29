import pytest
import torch
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.functional import (cross_correlation, fft_convolve,
                                 framewise_fir_filter, impulse_train,
                                 sinusoidal_harmonics,
                                 spectral_envelope_filter)
from tts_impl.net.vocoder.ddsp import SubtractiveVocoder


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_frames", [100, 200])
@pytest.mark.parametrize("post_filter_length", [0, 2048, 1024])
@pytest.mark.parametrize("n_fft", [1024])
@pytest.mark.parametrize("hop_length", [256])
@pytest.mark.parametrize("dim_periodicity", [80])
def test_subtractive_vocoder(
    batch_size: int,
    num_frames: int,
    post_filter_length: int,
    n_fft: int,
    hop_length: int,
    dim_periodicity: int,
):
    sample_rate = 24000
    vocoder = SubtractiveVocoder(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        dim_periodicity=dim_periodicity,
    )
    f0 = torch.ones(batch_size, num_frames) * 440.0
    per = torch.rand(batch_size, dim_periodicity, num_frames)
    env = torch.rand(batch_size, dim_periodicity, num_frames)
    pf = (
        torch.randn(batch_size, post_filter_length) if post_filter_length != 0 else None
    )
    o = vocoder.synthesize(f0, per, env, reverb=pf)
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
