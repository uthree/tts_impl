# HiFi-GAN Variants from https://arxiv.org/abs/2010.05646

from generator import HifiganGenerator
from discriminator import CombinedDiscriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator

from typing import List


class HifiganDiscriminator(CombinedDiscriminator):
    def __init__(
            self,
            periods: List[int] = [2, 3, 5, 7, 11],
            scales: List[int] =[1, 2, 4]
    ):
        super().__init__()
        self.sub_discriminators.append(MultiPeriodDiscriminator(periods))
        self.sub_discriminators.append(MultiScaleDiscriminator(scales))


class HifiganGeneratorV1(HifiganGenerator):
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
        super().__init__(
            input_channels,
            upsample_initial_channels,
            resblock_type,
            resblock_kernel_sizes,
            resblock_dilations,
            upsample_kernel_sizes,
            upsample_rates,
            output_channels
        )


class HifiganGeneratorV2(HifiganGenerator):
    def __init__(
            self,
            input_channels=80,
            upsample_initial_channels=128,
            resblock_type="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_kernel_sizes=[16, 16, 4, 4],
            upsample_rates=[8, 8, 2, 2],
            output_channels=1
    ):
        super().__init__(
            input_channels,
            upsample_initial_channels,
            resblock_type,
            resblock_kernel_sizes,
            resblock_dilations,
            upsample_kernel_sizes,
            upsample_rates,
            output_channels
        )


class HifiganGeneratorV3(HifiganGenerator):
    def __init__(
            self,
            input_channels=80,
            upsample_initial_channels=256,
            resblock_type="2",
            resblock_kernel_sizes=[3, 5, 7],
            resblock_dilations=[[1, 2], [2, 6], [3, 12]],
            upsample_kernel_sizes=[16, 16, 8],
            upsample_rates=[8, 8, 4],
            output_channels=1
    ):
        super().__init__(
            input_channels,
            upsample_initial_channels,
            resblock_type,
            resblock_kernel_sizes,
            resblock_dilations,
            upsample_kernel_sizes,
            upsample_rates,
            output_channels
        )