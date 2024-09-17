import os
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    # x: [BatchSize, cnannels, *]
    def forward(self, x: torch.Tensor):
        x = F.layer_norm(x.transpose(1, 2), (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, 2)
    

class ConvReluNorm(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel_size=3,
            dropout_rate=0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, 1, dropout_rate//2)
        self.norm = LayerNorm(output_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.norm(x)
        x = F.dropout(x, self.dropout_rate)
        return x


class VariancePredictor(torch.nn.Module):
    def __init__(
        self,
        input_channels=256,
        internal_channels=384,
        num_layers=1,
        kernel_size=3,
        dropout_rate=0.1,
        output_channels=1
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ConvReluNorm(input_channels, internal_channels, kernel_size, dropout_rate))
        for _ in range(num_layers):
            self.layers.append(ConvReluNorm(internal_channels, internal_channels, kernel_size, dropout_rate))
        self.layers.append(nn.Conv1d(internal_channels, 1, 1))


    def forward(self, x, x_mask):
        x = x * x_mask
        for layer in self.layers:
            x = layer(x) * x_mask
        return x


class VarianceAdapter(nn.Module):
    def __init__(self):
        pass