import pytest
import torch
from tts_impl.net.base.stateful import sanity_check_stateful_module
from tts_impl.net.common.normalization import EmaInstanceNorm, EmaLayerNorm


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
def test_ema_instancenorm(
    d_model, batch_size, seq_len, elementwise_affine, alpha_trainable
):
    norm = EmaInstanceNorm(
        d_model, elementwise_affine=elementwise_affine, alpha_trainable=alpha_trainable
    )
    x = torch.randn(batch_size, seq_len, d_model)
    y, h = norm(x)
    assert y.shape[0] == batch_size
    assert y.shape[1] == seq_len
    assert y.shape[2] == d_model


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [10, 20])
@pytest.mark.parametrize("elementwise_affine", [False, True])
@pytest.mark.parametrize("alpha_trainable", [False, True])
def test_ema_layernorm_sanity_check(
    d_model, batch_size, seq_len, elementwise_affine, alpha_trainable
):
    norm = EmaLayerNorm(
        d_model, elementwise_affine=elementwise_affine, alpha_trainable=alpha_trainable
    )
    x = torch.randn(batch_size, seq_len, d_model)
    sanity_check_stateful_module(norm, x)


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [10, 20])
@pytest.mark.parametrize("elementwise_affine", [False, True])
@pytest.mark.parametrize("alpha_trainable", [False, True])
def test_ema_instance_sanity_check(
    d_model, batch_size, seq_len, elementwise_affine, alpha_trainable
):
    norm = EmaInstanceNorm(
        d_model, elementwise_affine=elementwise_affine, alpha_trainable=alpha_trainable
    )
    x = torch.randn(batch_size, seq_len, d_model)
    sanity_check_stateful_module(norm, x)
