import json
from pathlib import Path

import torch

from .discriminator import (CombinedDiscriminator, MultiPeriodDiscriminator,
                            MultiScaleDiscriminator)
from .generator import HifiganGenerator


def load_discriminator_from_official_format(discriminator_path: Path, config_path: Path) -> CombinedDiscriminator:
    '''
    load discriminator pretrained parameters from official implementation's (https://github.com/jik876/hifi-gan)
    '''
    with open(config_path) as f:
        config = json.load(f)
    discriminator_dict = torch.load(discriminator_path, map_location='cpu')
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    mpd.load_state_dict(discriminator_dict['mpd'])
    msd.load_state_dict(discriminator_dict['msd'])
    discriminator = CombinedDiscriminator([mpd, msd])
    return discriminator


def load_generator_from_official_format(generator_path: Path, config_path: Path) -> HifiganGenerator:
    '''
    load generator pretrained parameters from official implementation's (https://github.com/jik876/hifi-gan)
    '''
    with open(config_path) as f:
        config = json.load(f)
    generator = HifiganGenerator(
        config['num_mels'],
        config['upsample_initial_channel'],
        config['resblock'],
        config['resblock_kernel_sizes'],
        config['resblock_dilation_sizes']
    )
    generator_dict = torch.load(generator_path, map_location='cpu')['generator']
    generator.load_state_dict(generator_dict)
    return generator

    