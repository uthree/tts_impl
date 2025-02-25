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


@pytest.mark.parametrize("activation", ["relu", "silu", "gelu"])
@pytest.mark.parametrize("glu", [True, False])
@pytest.mark.parametrize("rotary_pos_emb", [True, False])
@pytest.mark.parametrize("norm", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("prenorm", [True, False])
@pytest.mark.parametrize("window_size", [None, 4])
def test_vits_encoder(activation, glu, rotary_pos_emb, norm, prenorm, window_size):
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
    x = torch.randn(2, 192, 64)
    x_mask = torch.ones(2, 1, 64)
    o = enc(x, x_mask)


@pytest.mark.parametrize("activation", ["relu", "silu", "gelu"])
@pytest.mark.parametrize("glu", [True, False])
@pytest.mark.parametrize("rotary_pos_emb", [True, False])
@pytest.mark.parametrize("norm", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("prenorm", [True, False])
@pytest.mark.parametrize("window_size", [None, 4])
def test_vits_decoder(activation, glu, rotary_pos_emb, norm, prenorm, window_size):
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
    x = torch.randn(2, 192, 64)
    x_mask = torch.ones(2, 1, 64)
    h = torch.randn(2, 192, 32)
    h_mask = torch.ones(2, 1, 32)
    o = dec(x, x_mask, h, h_mask)
