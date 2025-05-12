import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.common.convnext import ConvNeXt1d


@pytest.mark.parametrize("in_channels", [80])
@pytest.mark.parametrize("out_channels", [128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("inter_channels", [192])
@pytest.mark.parametrize("ffn_channels", [256])
@pytest.mark.parametrize("num_layers", [2])
@pytest.mark.parametrize(
    "kernel_size",
    [
        3,
    ],
)
@pytest.mark.parametrize("grn", [True, False])
@pytest.mark.parametrize("glu", [True, False])
@pytest.mark.parametrize("norm", ["none", "layernorm", "tanh"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("input_length", [64, 100])
@pytest.mark.parametrize(
    "activation",
    [
        "gelu",
        "relu",
    ],
)
def test_convnext(
    in_channels,
    out_channels,
    batch_size,
    inter_channels,
    ffn_channels,
    num_layers,
    kernel_size,
    grn,
    glu,
    norm,
    causal,
    input_length,
    activation,
):
    model = ConvNeXt1d(
        in_channels=in_channels,
        out_channels=out_channels,
        inter_channels=inter_channels,
        ffn_channels=ffn_channels,
        kernel_size=kernel_size,
        num_layers=num_layers,
        grn=grn,
        glu=glu,
        norm=norm,
        causal=causal,
        activation=activation,
    )
    x = torch.randn(batch_size, in_channels, input_length)
    y = model(x)
    assert y.shape[0] == batch_size
    assert y.shape[1] == out_channels
    assert y.shape[2] == input_length
