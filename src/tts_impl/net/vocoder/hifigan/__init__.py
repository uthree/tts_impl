# HiFi-GAN Variants from https://arxiv.org/abs/2010.05646
import torch.optim as optim

import lightning as L
import torch.nn.functional as F

from .generator import HifiganGenerator
from .discriminator import CombinedDiscriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator

from tts_impl.net.vocoder.base import GanVocoderDiscriminator
from tts_impl.net.vocoder.hifigan.loss import feature_loss, generator_loss, discriminator_loss

import lightning as L

class Hifigan(L.LightningModule):
    def __init__(
            self,
            generator: HifiganGenerator,
            discriminator: GanVocoderDiscriminator,
        ):
        super().__init__()

        self.net_g = generator
        self.net_d = discriminator

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        waveform, input_features = batch
        opt_g, opt_d = self.optimizers()

        fake = self.net_g(input_features).sum(dim=1)

    def configure_optimizers(self):
        opt_g = optim.AdamW(self.generator.parameters(), lr=2e-4, betas=(0.8, 0.99))
        opt_d = optim.AdamW(self.discriminator.parameters(), lr=2e-4, betas=(0.8, 0.99))