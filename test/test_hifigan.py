import pytest
import torch
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.net.vocoder.discriminator import (
    CombinedDiscriminator,
    MultiPeriodDiscriminator,
    MultiResolutionStftDiscriminator,
    MultiScaleDiscriminator,
)
from tts_impl.net.vocoder.hifigan.lightning import (
    HifiganDiscriminator,
    HifiganGenerator,
)


@pytest.mark.parametrize("activation", ["lrelu", "silu", "gelu", "snake", "snakebeta"])
@pytest.mark.parametrize("alias_free", [True, False])
@pytest.mark.parametrize("resblock_type", ["1", "2"])
def test_hifigan_generator(activation, alias_free, resblock_type):
    G = HifiganGenerator(
        activation=activation, alias_free=alias_free, resblock_type=resblock_type
    )
    mel = torch.randn(2, 80, 32)
    wf = G(mel)
    assert wf.shape == torch.Size((2, 1, 8192))


def test_hifigan_discriminator():
    wf = torch.randn(2, 1, 8192)
    D = HifiganDiscriminator()
    logits, fmap = D(wf)


def test_custom_discriminator():
    wf = torch.randn(2, 1, 8192)
    D = CombinedDiscriminator(
        MultiPeriodDiscriminator([2, 3, 5, 7, 11]),
        MultiScaleDiscriminator([1, 2]),
        MultiResolutionStftDiscriminator([240, 120, 60]),
    )
    logits, fmap = D(wf)


def test_initialize_with_config():
    cfg = HifiganGenerator.Config()
    G = HifiganGenerator(**cfg)
