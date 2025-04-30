import pytest
import torch
from tts_impl.functional.monotonic_align import maximum_path


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("text_length", [100, 200])
@pytest.mark.parametrize("feat_length", [1000, 2000])
@pytest.mark.parametrize("algorithm", ["numba", "cython", "jit1", "jit2", "triton"])
def test_mas(batch_size: int, text_length: int, feat_length: int, algorithm: str):
    attn = torch.randn(batch_size, text_length, feat_length)
    attn_mask = torch.ones_like(attn)
    mpath = maximum_path(attn, attn_mask, algorithm=algorithm)
    assert mpath.shape == attn.shape
