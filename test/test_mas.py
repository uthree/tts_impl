from tts_impl.functional.monotonic_align import maximum_path
import torch

def test_mas():
    attn = torch.randn(2, 100, 1000)
    maximum_path(attn)