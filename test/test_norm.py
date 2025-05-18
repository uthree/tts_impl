import pytest
import torch
from tts_impl.net.common.normalization import EmaLayerNorm


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 10, 20])
@pytest.mark.parametrize("elementwise_affine", [False, True])
@pytest.mark.parametrize("alpha_trainable", [False, True])
def test_ema_layernorm(
    d_model, batch_size, seq_len, elementwise_affine, alpha_trainable
):
    norm = EmaLayerNorm(
        d_model, elementwise_affine=elementwise_affine, alpha_trainable=alpha_trainable
    )
    x = torch.randn(batch_size, seq_len, d_model)
    y, h = norm(x)
    assert y.shape[0] == batch_size
    assert y.shape[1] == seq_len
    assert y.shape[2] == d_model


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 10, 20])
@pytest.mark.parametrize("elementwise_affine", [False, True])
@pytest.mark.parametrize("alpha_trainable", [False, True])
def test_ema_layernorm_sanity_check(
    d_model, batch_size, seq_len, elementwise_affine, alpha_trainable
):
    norm = EmaLayerNorm(
        d_model, elementwise_affine=elementwise_affine, alpha_trainable=alpha_trainable
    )
    x = torch.randn(batch_size, seq_len, d_model)
    y_par, _ = norm(x)
    y_seq = []
    h_t = None
    for x_t in x.unbind(dim=1):
        x_t = x_t[:, None]
        y_t, h_t = norm(x_t, h_t)
        y_seq.append(y_t)
    y_seq = torch.cat(y_seq, dim=1)
    assert torch.allclose(y_par, y_seq, atol=1e-4)
