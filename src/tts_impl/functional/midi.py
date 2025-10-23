import math
from typing import Union

import torch
from torch.nn import functional as F


def freq2note(f0: Union[torch.Tensor, float], eps=1e-4) -> Union[torch.Tensor, float]:
    """
    Convert frequency to midi note scale.
    """
    if type(f0) == torch.Tensor:
        f0 = F.relu(f0) + eps
        n = torch.log2(f0 / 440.0) * 12.0 + 69.0
    else:
        f0 = f0 + eps
        n = math.log2(f0 / 440.0) * 12.0 + 69.0
    return n


def note2freq(n: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
    """
    Convert midi note to frequency.
    """
    return 440.0 * (2.0 ** ((n - 69.0) / 12.0))
