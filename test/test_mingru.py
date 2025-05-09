import pytest
import torch
from tts_impl.net.common.mingru import MinGRU


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("d_hidden", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 10])
def test_mingru(d_model, d_hidden, batch_size, seq_len):
    model = MinGRU(d_model=d_model, d_hidden=d_hidden)
    x = torch.randn(batch_size, seq_len, d_model)
    h = model(x)
    assert h.shape[0] == batch_size
    assert h.shape[1] == seq_len
    assert h.shape[2] == d_hidden


@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("d_hidden", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [10])
def test_mingru_sanity_check(d_model, d_hidden, batch_size, seq_len):
    model = MinGRU(d_model=d_model, d_hidden=d_hidden)
    x = torch.randn(batch_size, seq_len, d_model)
    h_par = model(x)
    h_seq = []
    h = None
    for x_t in x.unbind(dim=1):
        x_t = x_t[:, None]
        h = model(x_t, h)
        h_seq.append(h)
    h_seq = torch.cat(h_seq, dim=1)
    assert torch.allclose(h_par, h_seq, atol=1e-4)
