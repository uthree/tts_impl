# HiFi-GAN Variants from https://arxiv.org/abs/2010.05646

from .generator import HifiganGenerator
from .discriminator import CombinedDiscriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from .pretrained import from_pretrained