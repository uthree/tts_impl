# HiFi-GAN from https://arxiv.org/abs/2010.05646

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from tts_impl.net.base.vocoder import GanVocoderGenerator
from tts_impl.net.common.activation import init_activation
from tts_impl.utils.config import derive_config

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class LowPassFilter(nn.Module):
    def __init__(
        self,
        channels: int,
        cutoff: float = 0.5,
        kernel_size: int = 7,
    ):
        """
        cutoff: 遮断周波数 (サンプリング周波数に対する比率 0.0 ~ 1.0)
        kernel_size: カーネルサイズ (奇数を推奨)
        channels: 入力チャンネル数
        """
        super().__init__()

        if kernel_size % 2 == 0:
            kernel_size += 1  # 奇数に補正

        t = torch.linspace(-(kernel_size // 2), (kernel_size // 2), kernel_size)
        kernel = torch.sinc(cutoff * t)

        # ゲインを1に正規化
        kernel = kernel / (kernel.square().sum() + 1e-8).sqrt()

        # Conv1d用に形状変更: (out_channels, in_channels/groups, kernel_size)
        # depthwise convolutionを行うため、shapeは (channels, 1, kernel_size)
        self.register_buffer("weight", kernel.view(1, 1, -1).repeat(channels, 1, 1))

        self.kernel_size = kernel_size
        self.channels = channels

    def forward(self, x):
        # x shape: (batch_size, channels, length)
        # パディングしてサイズが変わらないようにする
        padding = self.kernel_size // 2
        x = F.pad(x, (padding, padding), mode="replicate")
        return F.conv1d(
            x,
            self.weight,
            groups=self.channels,
        )


class ResBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: list[int] = [1, 3, 5],
        activation: str = "lrelu",
        lowpass_filter: bool = False,
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.acts1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.acts2 = nn.ModuleList()
        if lowpass_filter:
            self.lowpass_filter = LowPassFilter(channels)
        else:
            self.lowpass_filter = nn.Identity()
        for d in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        get_padding(kernel_size, d),
                        d,
                    )
                )
            )
            self.acts1.append(init_activation(activation, channels))
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        get_padding(kernel_size, 1),
                        1,
                    )
                )
            )
            self.acts2.append(init_activation(activation, channels))

    def forward(self, x):
        for c1, c2, a1, a2 in zip(
            self.convs1, self.convs2, self.acts1, self.acts2, strict=False
        ):
            xt = self.lowpass_filter(x)
            xt = a1(xt)
            xt = c1(xt)
            xt = self.lowpass_filter(x)
            xt = a2(xt)
            xt = c2(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c1, c2 in zip(self.convs1, self.convs2, strict=False):
            remove_weight_norm(c1)
            remove_weight_norm(c2)


class ResBlock2(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: list[int] = [1, 3],
        activation: str = "lrelu",
        lowpass_filter: bool = False,
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.acts1 = nn.ModuleList()
        if lowpass_filter:
            self.lowpass_filter = LowPassFilter(channels)
        else:
            self.lowpass_filter = nn.Identity()
        for d in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        get_padding(kernel_size, d),
                        d,
                    )
                )
            )
            self.acts1.append(init_activation(activation, channels))

    def forward(self, x):
        for c1, a1 in zip(self.convs1, self.acts1, strict=False):
            xt = self.lowpass_filter(xt)
            xt = a1(xt)
            xt = c1(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c1 in self.convs1:
            remove_weight_norm(c1)


@derive_config
class HifiganGenerator(nn.Module, GanVocoderGenerator):
    """
    HiFi-GAN Generator purposed in https://arxiv.org/abs/2010.05646
    """

    def __init__(
        self,
        in_channels: int = 80,
        upsample_initial_channels: int = 512,
        resblock_type: str = "1",
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates: list[int] = [8, 8, 2, 2],
        lowpass_filter: bool = False,
        out_channels: int = 1,
        tanh_post_activation: bool = True,
        gin_channels: int = 0,
        activation: str = "lrelu",
        sample_rate: int = 22050,
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.frame_size = 1

        self.in_channels = in_channels
        self.upsample_initial_channels = upsample_initial_channels
        self.resblock_type = resblock_type
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilations = resblock_dilations
        self.upsample_rates = upsample_rates
        self.out_channels = out_channels
        self.tanh_post_activation = tanh_post_activation
        self.gin_channels = gin_channels
        self.sample_rate = sample_rate

        if resblock_type == "1":
            resblock = ResBlock1
        elif resblock_type == "2":
            resblock = ResBlock2
        else:
            raise ValueError("invalid resblock type")

        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channels, 7, 1, 3)
        )
        if gin_channels > 0:
            self.with_condition = True
            self.conv_cond = weight_norm(
                nn.Conv1d(gin_channels, upsample_initial_channels, 1, bias=False)
            )
        else:
            self.with_condition = False
            self.conv_cond = None

        self.ups = nn.ModuleList()
        self.up_lpfs = nn.ModuleList()
        self.up_acts = nn.ModuleList()
        for i, u in enumerate(upsample_rates):
            c1 = upsample_initial_channels // (2**i)
            c2 = upsample_initial_channels // (2 ** (i + 1))
            p = u // 2
            k = u * 2
            self.up_acts.append(init_activation(activation, channels=c1))
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, p)))
            lpf = (
                LowPassFilter(c2, cutoff=(1.0 / u), kernel_size=k)
                if lowpass_filter
                else nn.Identity()
            )
            self.up_lpfs.append(lpf)
            self.frame_size *= u

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channels // (2 ** (i + 1))
            for _j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilations, strict=False)
            ):
                self.resblocks.append(
                    resblock(
                        ch, k, d, activation=activation, lowpass_filter=lowpass_filter
                    )
                )

        self.post_act = init_activation(
            activation,
            channels=c2,
        )
        self.conv_post = weight_norm(nn.Conv1d(ch, out_channels, 7, 1, padding=3))
        self.lpf_post = LowPassFilter(ch) if lowpass_filter else nn.Identity()
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None, *args, **kwargs):
        """
        inputs:
            x: [batch_size, in_channels, num_frames], dtype=float
            g: [batch_size, 1, condition_dim] dtype=float, optional
        returns:
            waveform: [batch_size, out_channels, frames*frame_size]
                where: frame_size is the number of samples per frame.
        """
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.conv_cond(g)
        for i in range(self.num_upsamples):
            x = self.up_acts[i](x)
            x = self.ups[i](x)
            x = self.up_lpfs[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.lpf_post(x)
        x = self.post_act(x)
        x = self.conv_post(x)
        if self.tanh_post_activation:
            x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for b in self.resblocks:
            b.remove_weight_norm()
        if self.with_condition:
            remove_weight_norm(self.conv_cond)
