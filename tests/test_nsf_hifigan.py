import torch
import torch.nn as nn
import torch.nn.functional as F

from tts_impl.net.vocoder.nsf_hifigan import NsfhifiganGenerator


def test_nsf_hifigan_generator():
    G = NsfhifiganGenerator()
    mel = torch.randn(2, 80, 100)
    f0 = torch.rand(1, 1, 100)
    wf = G(mel, f0=f0)
    assert wf.shape == torch.Size((2, 1, 25600))
