import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.tts.nsf_vits import NsfvitsGenerator


@pytest.mark.parametrize("batch_size", [1, 4])
def test_nsf_vits_generator_forward(batch_size):
    G = NsfvitsGenerator()
    x = torch.randint(0, 255, (batch_size, 20))
    x_lengths = torch.IntTensor([20])
    y = torch.randn(batch_size, 80, 80)
    y_lengths = torch.IntTensor([80])
    f0 = torch.full((batch_size, 80), 440.0)
    G.forward(x, x_lengths, y, y_lengths, f0)
    o = G.infer(x, x_lengths)
