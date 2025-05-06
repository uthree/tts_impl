import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.common.convnext import ConvNeXt1d


@pytest.mark.parametrize("in_channels", [64, 128])
@pytest.mark.parametrize("out_channels", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("internal_channels", [64, 128])
@pytest.mark.parametrize("ffn_channels", [64, 128])
@pytest.mark.parametrize("num_layers", [1, 6])
@pytest.mark.parametrize("kernel_size", [3, 7])
@pytest.mark.parametrize("grn", [True, False])
@pytest.mark.parametrize("glu", [True, False])
@pytest.mark.parametrize("norm", ["none", "layernorm", "tanh"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("input_length", [64, 100])
def test_convnext(in_channels, out_channels, batch_size, internal_channels, ffn_channels, num_layers, kernel_size, grn, glu, norm, causal, input_length):
    model = ConvNeXt1d(in_channels, out_channels, internal_channels, ffn_channels, kernel_size, num_layers, grn, glu, norm, causal)
    x = torch.randn(batch_size, in_channels, input_length)
    y = model(x)
    assert y.shape[0] == batch_size
    assert y.shape[1] == out_channels
    assert y.shape[2] == input_length