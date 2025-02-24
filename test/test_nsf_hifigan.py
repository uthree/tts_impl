import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.vocoder.nsf_hifigan import NsfhifiganGenerator


@pytest.mark.parametrize("activation", ["lrelu", "silu", "gelu", "snake", "snakebeta"])
@pytest.mark.parametrize("alias_free", [True, False])
def test_nsf_hifigan_generator(activation, alias_free):
    G = NsfhifiganGenerator(filter={"activation": activation, "alias_free": alias_free})
    mel = torch.randn(2, 80, 100)
    f0 = torch.rand(1, 100)
    wf = G(mel, f0=f0)
    assert wf.shape == torch.Size((2, 1, 25600))


def test_initialize_with_config():
    cfg = NsfhifiganGenerator.Config()
    G = NsfhifiganGenerator(**cfg)
