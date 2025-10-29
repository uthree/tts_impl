import pytest
import torch
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.net.tts.vits.lightning import VitsGenerator


def test_vits_generator():
    G = VitsGenerator()
    x = torch.randint(0, 255, (1, 20))
    x_lengths = torch.IntTensor([20])
    y = torch.randn(1, 80, 80)
    y_lengths = torch.IntTensor([80])
    G.forward(x, x_lengths, y, y_lengths)
    o = G.infer(x, x_lengths)


from tts_impl.net.tts.vits.attentions import Decoder, Encoder
from tts_impl.net.tts.vits.lightning import VitsLightningModule
from tts_impl.net.tts.vits.models import (DurationPredictor, PosteriorEncoder,
                                          ResidualCouplingBlock,
                                          StochasticDurationPredictor,
                                          TextEncoder)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("length", [100, 200])
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("glu", [True, False])
@pytest.mark.parametrize("rotary_pos_emb", [True, False])
@pytest.mark.parametrize("norm", ["rmsnorm"])
@pytest.mark.parametrize("prenorm", [True, False])
@pytest.mark.parametrize("window_size", [None, 4])
@pytest.mark.parametrize("hidden_dim", [192])
def test_vits_encoder(
    batch_size,
    length,
    activation,
    glu,
    rotary_pos_emb,
    norm,
    prenorm,
    window_size,
    hidden_dim,
):
    enc = Encoder(
        hidden_dim,
        768,
        3,
        2,
        3,
        0,
        window_size=window_size,
        norm=norm,
        prenorm=prenorm,
        glu=glu,
        rotary_pos_emb=rotary_pos_emb,
        activation=activation,
    )
    x = torch.randn(batch_size, hidden_dim, length)
    x_mask = torch.ones(batch_size, 1, length)
    o = enc.forward(x, x_mask)
    assert o.shape[0] == batch_size
    assert o.shape[1] == hidden_dim
    assert o.shape[2] == length


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("x_length", [100])
@pytest.mark.parametrize("h_length", [200])
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("glu", [True, False])
@pytest.mark.parametrize("rotary_pos_emb", [True, False])
@pytest.mark.parametrize("norm", ["rmsnorm"])
@pytest.mark.parametrize("prenorm", [True, False])
@pytest.mark.parametrize("window_size", [None, 4])
@pytest.mark.parametrize("hidden_dim", [192])
def test_vits_decoder(
    batch_size,
    x_length,
    h_length,
    activation,
    glu,
    rotary_pos_emb,
    norm,
    prenorm,
    window_size,
    hidden_dim,
):
    dec = Decoder(
        hidden_dim,
        768,
        3,
        2,
        3,
        0,
        window_size=window_size,
        norm=norm,
        prenorm=prenorm,
        glu=glu,
        rotary_pos_emb=rotary_pos_emb,
        activation=activation,
    )
    x = torch.randn(batch_size, hidden_dim, x_length)
    x_mask = torch.ones(batch_size, 1, x_length)
    h = torch.randn(batch_size, hidden_dim, h_length)
    h_mask = torch.ones(batch_size, 1, h_length)
    o = dec(x, x_mask, h, h_mask)
    assert o.shape[0] == batch_size
    assert o.shape[1] == hidden_dim
    assert o.shape[2] == x_length


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("length", [100, 200])
@pytest.mark.parametrize("in_channels", [40])
@pytest.mark.parametrize("hidden_channels", [64])
@pytest.mark.parametrize("out_channels", [192])
@pytest.mark.parametrize("dilation_rate", [1])
@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("gin_channels", [0, 192])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_posterior_encoder(
    batch_size,
    length,
    in_channels,
    out_channels,
    hidden_channels,
    kernel_size,
    gin_channels,
    dilation_rate,
    n_layers,
):
    posterior_encoder = PosteriorEncoder(
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=gin_channels,
    )
    x = torch.randn(batch_size, in_channels, length)
    x_lengths = torch.randint(low=1, high=length, size=(batch_size,))
    g = torch.randn(batch_size, gin_channels, 1) if gin_channels > 0 else None
    z_q, m_q, logs_q, z_mask = posterior_encoder.forward(x, x_lengths, g=g)
    assert z_q.shape[0] == batch_size
    assert z_q.shape[1] == out_channels
    assert z_q.shape[2] == length


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("hidden_channels", [128])
@pytest.mark.parametrize("channels", [192])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("dilation_rate", [1])
@pytest.mark.parametrize("n_layers", [2])
@pytest.mark.parametrize("n_flows", [2])
@pytest.mark.parametrize("gin_channels", [0, 192])
@pytest.mark.parametrize("reverse", [False, True])
def test_flow(
    batch_size,
    length,
    hidden_channels,
    channels,
    kernel_size,
    dilation_rate,
    n_layers,
    n_flows,
    gin_channels,
    reverse,
):
    flow = ResidualCouplingBlock(
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows,
        gin_channels,
    )
    z = torch.randn(batch_size, channels, length)
    z_mask = torch.ones(batch_size, 1, length)
    g = torch.randn(batch_size, gin_channels, 1) if gin_channels > 0 else None
    o = flow.forward(z, z_mask, g=g, reverse=reverse)
    assert o.shape == z.shape


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("in_channels", [192])
@pytest.mark.parametrize("filter_channels", [192])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("gin_channels", [0, 192])
def test_duration_predictor(
    batch_size,
    length,
    in_channels,
    filter_channels,
    kernel_size,
    p_dropout,
    gin_channels,
):
    duration_predictor = DurationPredictor(
        in_channels, filter_channels, kernel_size, p_dropout, gin_channels=gin_channels
    )
    x = torch.randn(batch_size, in_channels, length)
    x_mask = torch.ones(batch_size, 1, length)
    g = torch.randn(batch_size, gin_channels, 1) if gin_channels > 0 else None
    o = duration_predictor.forward(x, x_mask, g=g)
    assert o.shape[0] == batch_size
    assert o.shape[1] == 1
    assert o.shape[2] == length


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("in_channels", [192])
@pytest.mark.parametrize("out_channels", [1, 4])
@pytest.mark.parametrize("filter_channels", [128])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("p_dropout", [0.1])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("n_flows", [1, 2])
@pytest.mark.parametrize("gin_channels", [0, 192])
@pytest.mark.parametrize("condition_backward", [True, False])
def test_stochastic_duration_predictor(
    batch_size,
    length,
    in_channels,
    out_channels,
    filter_channels,
    kernel_size,
    p_dropout,
    reverse,
    gin_channels,
    n_flows,
    condition_backward,
):
    stochastic_duration_predictor = StochasticDurationPredictor(
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows,
        gin_channels,
        condition_backward,
    )
    x = torch.randn(batch_size, in_channels, length)
    x_mask = torch.ones(batch_size, 1, length)
    g = torch.randn(batch_size, gin_channels, 1) if gin_channels > 0 else None
    w = torch.randint(low=1, high=10, size=(batch_size, out_channels, length)).float()
    o = stochastic_duration_predictor.forward(x, x_mask, g=g, w=w, reverse=reverse)
    assert o.shape[0] == batch_size
    if reverse:
        assert o.shape[1] == out_channels
    else:
        o.mean().backward()
