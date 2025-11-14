import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.vocoder.hifigan import HifiganDiscriminator
from .generator import Generator
from tts_impl.utils.config import derive_config
from tts_impl.net.tts.vits.commons import slice_segments
from tts_impl.net.tts.vits.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
)

_vits_discriminator_config = HifiganDiscriminator.Config()
_vits_discriminator_config.msd.scales = [1]
_vits_discriminator_config.mpd.periods = [2, 3, 5, 7, 11]
_vits_discriminator_config.mrsd.n_fft = [240, 400, 600]
_vits_discriminator_config.mrsd.hop_size = [50, 100, 200]

@derive_config
class NhvcLightningModule:
    pass