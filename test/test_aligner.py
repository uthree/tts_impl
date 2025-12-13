import pytest
import torch

from tts_impl.net.aligner import ForcedAligner


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [16, 32])
@pytest.mark.parametrize("text_length", [100])
@pytest.mark.parametrize("feat_length", [1000])
def test_alginer(batch_size: int, channels: int, text_length: int, feat_length: int):
    feat = torch.randn(batch_size, channels, feat_length)
    text = torch.randn(batch_size, channels, text_length)
    aligner = ForcedAligner(channels, channels)
    dur, loss = aligner.align(
        text,
        feat,
        torch.LongTensor([text_length] * batch_size),
        torch.LongTensor([feat_length] * batch_size),
    )
    assert dur.shape[0] == batch_size
