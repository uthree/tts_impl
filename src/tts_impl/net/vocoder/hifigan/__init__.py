import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tts_impl.net.vocoder.base import GanVocoderDiscriminator, GanVocoderGenerator, GanVocoder
from tts_impl.net.vocoder.hifigan.loss import feature_loss, generator_loss, discriminator_loss

from typing import Optional


# HiFi-GAN from https://arxiv.org/abs/2010.05646
class Hifigan(GanVocoder):
    def __init__(
            self,
            generator: GanVocoderGenerator,
            discriminator: GanVocoderDiscriminator,
            spectrogram_extractor: nn.Module
        ):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.spectrogram = spectrogram_extractor

        self.automatic_optimization = False

    def training_step(self, batch):
        waveform, input_features = batch
        spec_real = self.spectrogram(waveform.sum(1)).detach()
        opt_g, opt_d = self.optimizers()

        # Train G.
        fake = self.generator(input_features)
        logits, fmap_fake = self.generator(fake)
        _, fmap_real = self.discriminator(waveform)
        loss_adv = generator_loss(logits)
        loss_feat = feature_loss(fmap_real, fmap_fake)
        spec_fake = self.spectrogram(fake.sum(1))
        loss_mel = F.l1_loss(spec_fake, spec_real)
        loss_g = loss_mel * 45.0 + loss_feat + loss_adv

        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        self.clip_gradients(opt_g, 1.0, "norm")
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # Train D.
        fake = fake.detach()
        logits_fake = self.discriminator(fake)
        logits_real = self.discriminator(waveform)
        loss_d = discriminator_loss(logits_real, logits_fake)

        self.toggle_optimizer(opt_d)
        self.manual_backward(loss_d)
        self.clip_gradients(opt_d, 1.0, "norm")
        opt_d.step()
        self.untoggle_optimizer(opt_d)
        
        # Logs
        self.log("loss/Mel Spectrogram", loss_mel.item())
        self.log("loss/Feature Matching", loss_feat.item())
        self.log("loss/Generator Adversarial", loss_adv.item())
        self.log("loss/Discriminator Adversarial", loss_d.item())

    def configure_optimizers(self):
        opt_g = optim.AdamW(self.generator.parameters(), lr=2e-4, betas=(0.8, 0.99))
        opt_d = optim.AdamW(self.discriminator.parameters(), lr=2e-4, betas=(0.8, 0.99))
        return opt_g, opt_d