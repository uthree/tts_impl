import pytest
import torch
from tts_impl.net.common.normalization import EmaLayerNorm


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 10, 20])
def test_ema_layernorm(d_model, batch_size, seq_len):
    norm = EmaLayerNorm(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    x, h = EmaLayerNorm.forward(x)
    assert h.shape[0] == batch_size
    assert h.shape[1] == seq_len
    assert h.shape[2] == d_model
