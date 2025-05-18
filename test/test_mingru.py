import pytest
import torch
from tts_impl.net.common.mingru import MinGRU


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("d_hidden", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 10, 20])
def test_mingru(d_model, d_hidden, batch_size, seq_len):
    model = MinGRU(d_model=d_model, d_hidden=d_hidden)
    x = torch.randn(batch_size, seq_len, d_model)
    y, _ = model(x)
    assert y.shape[0] == batch_size
    assert y.shape[1] == seq_len
    assert y.shape[2] == d_hidden


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("d_hidden", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [10, 20])
def test_mingru_sanity_check(d_model, d_hidden, batch_size, seq_len):
    model = MinGRU(d_model=d_model, d_hidden=d_hidden)
    x = torch.randn(batch_size, seq_len, d_model)
    y_par, _ = model(x)
    y_seq = []
    h_t = None
    for x_t in x.unbind(dim=1):
        x_t = x_t[:, None]
        y_t, h_t = model(x_t, h_t)
        y_seq.append(y_t)
    y_seq = torch.cat(y_seq, dim=1)
    assert torch.allclose(y_par, y_seq, atol=1e-4)
