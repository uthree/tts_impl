import torch
import torch.nn.functional as F
from tts_impl.net.aligner import ForcedAligner


def test_alginer():
    feat = torch.randn(2, 10, 1000)
    text = torch.randn(2, 10, 100)
    aligner = ForcedAligner(10, 10)
    dur, loss = aligner.align(
        text, feat, torch.LongTensor([100, 100]), torch.LongTensor([1000, 1000])
    )
