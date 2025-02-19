import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.tts.vits.lightning import VitsGenerator


def test_vits_generator():
    G = VitsGenerator()
    x = torch.randint(0, 255, (1, 20))
    x_lengths = torch.IntTensor([20])
    y = torch.randn(1, 80, 80)
    y_lengths = torch.IntTensor([80])
    G.forward(x, x_lengths, y, y_lengths)
    o = G.infer(x, x_lengths)
