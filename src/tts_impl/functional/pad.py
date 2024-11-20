import torch
import torch.nn as nn
import torch.nn.functional as F


def adjust_size_1d(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[2] < size:
        pad_size = size - x.shape[2]
        x = F.pad(x, (0, pad_size))
    if x.shape[2] > size:
        x = x[:, :, size]
    return x
