import torch
import torch.nn as nn
import torch.nn.functional as F

from tts_impl.net.vocoder.hifigan import HifiganGenerator, CombinedDiscriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator

def test_hifigan_generator():
    G = HifiganGenerator()
    mel = torch.randn(2, 80, 100)
    wf = G(mel)
    assert wf.shape == torch.Size((2, 1, 25600))

def test_hifigan_discriminator():
    wf = torch.randn(2, 1, 10000)
    D = CombinedDiscriminator([MultiPeriodDiscriminator(), MultiScaleDiscriminator()])
    logits, fmap = D(wf)