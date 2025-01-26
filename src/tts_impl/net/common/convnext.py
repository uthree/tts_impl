import torch
import torch.nn as nn
import torch.nn.functional as F


# Layer normalization
class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    # x: [BatchSize, cnannels, *]
    def forward(self, x: torch.Tensor):
        x = F.layer_norm(
            x.transpose(1, 2), (self.channels,), self.gamma, self.beta, self.eps
        )
        return x.transpose(1, 2)


# Global Resnponse Normalization for 1d Sequence (shape=[BatchSize, Channels, Length])
class GRN1d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    # x: [batchsize, channels, length]
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class DepthwiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, causal=False):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=self.channels)
        if causal:
            self.pad = nn.ReflectionPad1d((0, self.kernel_size - 1))
        else:
            self.pad = nn.ReflectionPad1d(
                (self.kernel_size // 2, self.kernel_size // 2)
            )

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x
