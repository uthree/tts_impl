import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tts_impl.net.vocoder.hifigan import HifiganGeneratorV1

def test_hifigan_generator():
    G = HifiganGeneratorV1()
    mel = torch.randn(2, 80, 100)
    wf = G(mel)
    assert wf.shape == torch.Size((2, 1, 25600))