import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.c1 = nn.Conv1d(
            channels, channels, kernel_size, 1, kernel_size // 2, dilation=dilation
        )
        self.c2 = nn.Conv1d(channels, channels, 1, dilation=dilation)

    def forward(self, x):
        res = x
        x = F.gelu(x)
        x = self.c1(x)
        x = F.gelu(x)
        x = self.c2(x)
        return x + res
