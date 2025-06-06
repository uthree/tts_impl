import torch
import torch.nn as nn
import torch.nn.functional as F


def estimate_energy(waveform: torch.Tensor, hop_length: int) -> torch.Tensor:
    """
    Args:
        waveform: (batch_size, channels, length)

    Returns:
        energy: (batch_size, channels, length / hop_length)
    """
    return F.max_pool1d(waveform.abs(), hop_length)
