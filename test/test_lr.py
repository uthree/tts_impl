import torch
from torch import nn as nn
from torch.nn import functional as F
from tts_impl.functional.length_regurator import (
    duplicate_by_duration,
    gaussian_upsampling,
)


def test_length_regurator():
    x = torch.randn(2, 192, 100)
    w = torch.randint(1, 10, (2, 100))
    out = duplicate_by_duration(x, w)
    assert out.shape[2] == w.sum(dim=1).max()


def test_gaussian_upsampling():
    x = torch.randn(2, 192, 100)
    w = torch.randint(1, 10, (2, 100))
    out = gaussian_upsampling(x, w)
    assert out.shape[2] == w.sum(dim=1).max()
