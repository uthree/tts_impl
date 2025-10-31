import pytest
import torch
from tts_impl.net.base.stateful import sanity_check_stateful_module
from tts_impl.net.common.causal_conv import CachedCausalConv


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("kernel_size", [2, 4])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 10, 20])
@pytest.mark.parametrize("bias", [False, True])
def test_dwc_conv(d_model, kernel_size, batch_size, seq_len, bias):
    model = CachedCausalConv(d_model, kernel_size, bias=bias)
    x = torch.randn(batch_size, seq_len, d_model)
    y, _ = model(x)
    assert y.shape[0] == batch_size
    assert y.shape[1] == seq_len
    assert y.shape[2] == d_model


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("kernel_size", [2, 4])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [10, 20])
@pytest.mark.parametrize("bias", [False, True])
def test_mingru_sanity_check(d_model, kernel_size, batch_size, seq_len, bias):
    model = CachedCausalConv(d_model, kernel_size, bias=bias)
    x = torch.randn(batch_size, seq_len, d_model)
    y_par, _ = model(x)
    sanity_check_stateful_module(model, x)
