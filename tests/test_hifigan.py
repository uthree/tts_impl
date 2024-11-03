import torch
import torch.nn as nn
import torch.nn.functional as F

from tts_impl.net.vocoder.hifigan import HifiganDiscriminator, HifiganGenerator
from tts_impl.net.vocoder.discriminator import CombinedDiscriminator, MultiPeriodDiscriminator, MultiResolutionStftDiscriminator, MultiScaleDiscriminator


def test_hifigan_generator():
    G = HifiganGenerator()
    mel = torch.randn(2, 80, 100)
    wf = G(mel)
    assert wf.shape == torch.Size((2, 1, 25600))


def test_hifigan_discriminator():
    wf = torch.randn(2, 1, 10000)
    D = HifiganDiscriminator()
    logits, fmap = D(wf)


def test_custom_discriminator():
    wf = torch.randn(2, 1, 10000)
    D = CombinedDiscriminator(
        MultiPeriodDiscriminator([2, 3, 5, 7, 11]),
        MultiScaleDiscriminator([1, 2]),
        MultiResolutionStftDiscriminator([240, 120, 60])
    )
    logits, fmap = D(wf)