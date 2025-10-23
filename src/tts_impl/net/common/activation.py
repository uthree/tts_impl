from typing import Literal, TypeAlias

import torch
from alias_free_torch import Activation1d as AliasFreeActivation1d
from torch import nn as nn
from torch.nn import functional as F

ActivationName: TypeAlias = Literal[
    "lrelu", "silu", "gelu", "snake", "snakebeta", "linear", "elu"
]
LRELU_SLOPE = 0.1


class Snake(nn.Module):
    def __init__(
        self,
        channels: int,
        alpha: float = 0.0,
        trainable: bool = True,
        logscale: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.logscale = logscale
        self.eps = eps

        if logscale:
            self.alpha = nn.Parameter(torch.zeros(channels) + alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(channels) * alpha)

        self.alpha.requires_grad = trainable

    def forward(self, x):
        alpha = self.alpha[None, :, None]
        if self.logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.eps)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class SnakeBeta(nn.Module):
    def __init__(
        self,
        channels: int,
        alpha: float = 0.0,
        beta: float = 0.0,
        trainable: bool = True,
        logscale: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.logscale = logscale
        self.eps = eps

        if logscale:
            self.alpha = nn.Parameter(torch.zeros(channels) + alpha)
            self.beta = nn.Parameter(torch.zeros(channels) + beta)
        else:
            self.alpha = nn.Parameter(torch.ones(channels) * alpha)
            self.beta = nn.Parameter(torch.ones(channels) * beta)

        self.alpha.requires_grad = trainable
        self.beta.requires_grad = trainable

    def forward(self, x):
        alpha = self.alpha[None, :, None]
        beta = self.beta[None, :, None]
        if self.logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.eps)) * torch.pow(torch.sin(x * alpha), 2)
        return x


def init_activation(
    name: ActivationName = "lrelu", channels: int = 0, alias_free: bool = False
) -> nn.Module:
    if alias_free:
        act = init_activation(name, channels=channels, alias_free=False)
        act = AliasFreeActivation1d(act)
        return act
    if name == "relu":
        return nn.ReLU()
    elif name == "lrelu":
        return nn.LeakyReLU(LRELU_SLOPE)
    elif name == "silu" or name == "swish":
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "snake":
        return Snake(channels=channels)
    elif name == "snakebeta":
        return SnakeBeta(channels=channels)
    elif name == "elu":
        return nn.ELU()
    elif name == "softplus":
        return nn.Softplus()
    elif name == "linear" or name == "identity":
        return nn.Identity()
    else:
        raise ValueError(
            f"Invalid activation name: `{name}`; available: lrelu, relu, gelu, elu, silu, snake, snakebeta, softpuls, linear"
        )
