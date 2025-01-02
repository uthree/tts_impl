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

    # x: [BatchSize, channels, *]
    def forward(self, x: torch.Tensor):
        x = F.layer_norm(
            x.transpose(1, 2), (self.channels,), self.gamma, self.beta, self.eps
        )
        x = x.transpose(1, 2)
        return x


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


class Convnext1dLayer(nn.Module):
    """
    ConvNeXt v2 Layer
    """

    def __init__(
        self, channels: int, kernel_size: int, mlp_mul: int, causal: bool = False
    ):
        super().__init__()
        if causal:
            self.pad = nn.ZeroPad1d((0, kernel_size - 1))
        else:
            self.pad = nn.ZeroPad1d((kernel_size // 2, kernel_size // 2))
        self.c1 = nn.Conv1d(channels, channels, kernel_size, 1, 0, groups=channels)
        self.norm = LayerNorm1d(channels)
        self.c2 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.grn = GRN1d(channels * mlp_mul)
        self.c3 = nn.Conv1d(channels * mlp_mul, channels, 1)

    def forward(self, x):
        """
        Args:
            x: [batchsize, channels, length]
        Returns:
            x: [batchsize, channels, length]
        """
        res = x
        x = self.pad(x)
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.gelu(x)
        x = self.grn(x)
        x = self.c3(x)
        x = x + res
        return x


class Convnext1d(nn.Module):
    """
    ConvNeXt v2
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        mlp_mul: int,
        num_layers: int,
        causal: bool,
    ):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels)
        self.layers = nn.Sequential(
            *[
                Convnext1dLayer(hidden_channels, kernel_size, mlp_mul, causal)
                for _ in range(num_layers)
            ]
        )
        self.post_norm = LayerNorm1d(hidden_channels)
        self.post = nn.Conv1d(hidden_channels, out_channels)

    def forward(self, x):
        """
        Args:
            x: [batchsize, in_channels, length]
        Returns:
            x: [batchsize, out_channels, length]
        """
        x = self.pre(x)
        x = self.layers(x)
        x = self.post_norm(x)
        x = self.post(x)
        return x
