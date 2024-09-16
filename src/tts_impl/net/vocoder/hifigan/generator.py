# HiFi-GAN from https://arxiv.org/abs/2010.05646

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm
from utils import get_padding


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, d), d)))
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, 1), 1)))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt
        return x
    

class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, d), d)))

    def forward(self, x):
        for c1 in self.convs1:
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            x = x + xt
        return x
    

class HifiganGenerator(nn.Module):
    def __init__(
            self,
            input_channels=80,
            upsample_initial_channels=512,
            resblock_type="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_kernel_sizes=[16, 16, 4, 4],
            upsample_rates=[8, 8, 2, 2],
            output_channels=1
        ):

        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        if resblock_type == "1":
            resblock = ResBlock1
        elif resblock_type == "2":
            resblock = ResBlock2
        else:
            raise "invalid resblock type"

        self.conv_pre = weight_norm(nn.Conv1d(input_channels, upsample_initial_channels, 7, 1, 3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c1 = upsample_initial_channels//(2**i)
            c2 = upsample_initial_channels//(2**(i+1))
            p = (k-u)//2
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, p)))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channels//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilations)):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = weight_norm(nn.Conv1d(ch, output_channels, 7, 1, padding=3))

        self.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            x = F.leaky_relu(x, 0.1)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
