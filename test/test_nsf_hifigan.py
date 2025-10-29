import pytest
import torch
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.net.vocoder.nsf_hifigan import NsfhifiganGenerator


@pytest.mark.parametrize("activation", ["lrelu", "silu", "gelu", "snake", "snakebeta"])
@pytest.mark.parametrize("alias_free", [True, False])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("in_channels", [40, 80])
@pytest.mark.parametrize("num_frames", [32, 16])
def test_nsf_hifigan_generator(
    activation, alias_free, batch_size, in_channels, num_frames
):
    frame_size = 256
    G = NsfhifiganGenerator(
        filter_module={
            "activation": activation,
            "alias_free": alias_free,
            "in_channels": in_channels,
        }
    )
    mel = torch.randn(batch_size, in_channels, num_frames)
    f0 = torch.exp(torch.rand(batch_size, num_frames) * 5)
    wf = G(mel, f0=f0)
    assert wf.shape == torch.Size((batch_size, 1, num_frames * frame_size))


def test_initialize_with_config():
    cfg = NsfhifiganGenerator.Config()
    G = NsfhifiganGenerator(**cfg)
