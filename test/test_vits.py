import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
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


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("length", [100, 200])
@pytest.mark.parametrize("activation", ["relu", "silu", "gelu"])
@pytest.mark.parametrize("glu", [True, False])
@pytest.mark.parametrize("rotary_pos_emb", [True, False])
@pytest.mark.parametrize("norm", ["rmsnorm", "layernorm", "none", "tanh"])
@pytest.mark.parametrize("prenorm", [True, False])
@pytest.mark.parametrize("window_size", [None, 4])
def test_vits_encoder(
    batch_size, length, activation, glu, rotary_pos_emb, norm, prenorm, window_size
):
    enc = Encoder(
        192,
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
    x = torch.randn(batch_size, 192, length)
    x_mask = torch.ones(batch_size, 1, length)
    o = enc.forward(x, x_mask)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("x_length", [100, 200])
@pytest.mark.parametrize("h_length", [100, 200])
@pytest.mark.parametrize("activation", ["relu", "silu", "gelu"])
@pytest.mark.parametrize("glu", [True, False])
@pytest.mark.parametrize("rotary_pos_emb", [True, False])
@pytest.mark.parametrize("norm", ["rmsnorm", "layernorm", "none", "tanh"])
@pytest.mark.parametrize("prenorm", [True, False])
@pytest.mark.parametrize("window_size", [None, 4])
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
):
    dec = Decoder(
        192,
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
    x = torch.randn(batch_size, 192, x_length)
    x_mask = torch.ones(batch_size, 1, x_length)
    h = torch.randn(batch_size, 192, h_length)
    h_mask = torch.ones(batch_size, 1, h_length)
    o = dec(x, x_mask, h, h_mask)
