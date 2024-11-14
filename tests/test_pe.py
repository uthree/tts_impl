import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.functional import estimate_f0


@pytest.mark.parametrize("algorithm", ["dio", "harvest", "fcpe"])
def test_f0_estimation(algorithm):
    wf = torch.randn(2, 10000)
    estimate_f0(wf, 24000, 480, algorithm)
