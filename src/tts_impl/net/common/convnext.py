import torch
import torch.nn as nn
import torch.nn.functional as F


# Layer normalization
class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-6):
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
class GlobalResponseNorm1d(nn.Module):
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
    def __init__(self, channels, kernel_size, causal=False, norm=True):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=self.channels)
        if norm:
            self.norm = LayerNorm1d(channels)
        else:
            self.norm = nn.Identity()
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


class FeedForward1d(nn.Module):
    def __init__(self, channels, internal_channels, grn=True, glu=False):
        super().__init__()
        self.glu = glu
        if glu:
            self.c1 = nn.Conv1d(channels, internal_channels * 2, 1)
        else:
            self.c2 = nn.Conv1d(channels, internal_channels, 1)
        self.c2 = nn.Conv1d(internal_channels, channels, 1)
        if grn:
            self.grn = GlobalResponseNorm1d(internal_channels)
        else:
            self.grn = nn.Identity()

    def forward(self, x):
        if self.glu:
            x = self.c1(x)
            x_0, x_1 = x.chunk(2, dim=1)
            x = x_0 * F.gelu(x_1)
        else:
            x = self.c1(x)
            x = F.gelu(x)
        x = self.grn(x)
        x = self.c2(x)
        return x


class ConvNeXtLayer(nn.Module):
    def __init__(
        self,
        channels,
        ffn_channels,
        kernel_size=7,
        grn=True,
        glu=False,
        norm=True,
        causal=False,
    ):
        super().__init__()
        self.dw = DepthwiseConv1d(channels, kernel_size, causal, norm)
        self.ffn = FeedForward1d(channels, ffn_channels, grn, glu)

    def forward(self, x):
        res = x
        x = self.dw(x)
        x = self.ffn(x)
        x = res + x
        return x


class ConvNexXtStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inter_channels: int,
        ffn_channels: int,
        num_layers: int,
        kernel_size: int = 7,
        grn: bool = True,
        glu: bool = False,
        norm: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, inter_channels, 1)
        self.layers = nn.Sequential(
            *[
                ConvNeXtLayer(
                    inter_channels, ffn_channels, kernel_size, grn, glu, norm, causal
                )
                for _ in range(num_layers)
            ]
        )
        self.post_norm = LayerNorm1d(inter_channels) if norm else nn.Identity()
        self.out_conv = nn.Conv1d(inter_channels, out_channels, 1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layers(x)
        x = self.post_norm(x)
        x = self.out_conv(x)
        return x
