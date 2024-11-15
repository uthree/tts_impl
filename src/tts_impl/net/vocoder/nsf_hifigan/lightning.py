import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig

from tts_impl.net.vocoder.hifigan import HifiganDiscriminator
from tts_impl.net.vocoder.hifigan.loss import (discriminator_loss,
                                               feature_loss, generator_loss)
from tts_impl.transforms import LogMelSpectrogram

from .generator import NsfhifiganGenerator


# HiFi-GAN from https://arxiv.org/abs/2010.05646
class HifiganLightningModule(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.automatic_optimization = False

        self.generator = NsfhifiganGenerator(**config.generator)
        self.discriminator = HifiganDiscriminator(**config.discriminator)
        self.melspectrogram_extractor = LogMelSpectrogram(**config.mel)

        self.save_hyperparameters()

    def training_step(self, batch):
        waveform, data = batch

        acoustic_features = data["acoustic_features"]
        f0 = data["f0"]

        spec_real = self.spectrogram(waveform.sum(1)).detach()
        opt_g, opt_d = self.optimizers()

        weight_mel = self.config.get("weight_mel", 45.0)
        weight_feat = self.config.get("weight_feat", 1.0)
        weight_adv = self.config.get("weight_adv", 1.0)

        # Train G.
        fake = self.generator(acoustic_features, f0=f0)
        logits, fmap_fake = self.generator(fake)
        _, fmap_real = self.discriminator(waveform)
        loss_adv, loss_adv_list = generator_loss(logits)
        loss_feat = feature_loss(fmap_real, fmap_fake)
        spec_fake = self.spectrogram(fake.sum(1))
        loss_mel = F.l1_loss(spec_fake, spec_real)
        loss_g = loss_mel * weight_mel + loss_feat * weight_feat + loss_adv * weight_adv

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
        loss_d, loss_d_list_r, loss_d_list_f = discriminator_loss(
            logits_real, logits_fake
        )

        self.toggle_optimizer(opt_d)
        self.manual_backward(loss_d)
        self.clip_gradients(opt_d, 1.0, "norm")
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # Logs
        self.log("loss/Generator All", loss_g.item())
        self.log("loss/Mel Spectrogram", loss_mel.item())
        self.log("loss/Feature Matching", loss_feat.item())
        self.log("loss/Generator Adversarial", loss_adv.item())
        self.log("loss/Discriminator Adversarial", loss_d.item())

        for i, l in enumerate(loss_adv_list):
            self.log(f"Generator Adversarial/{i}", l.item())
        for i, l in enumerate(loss_d_list_f):
            self.log(f"Discriminator Adversarial/fake {i}", l.item())
        for i, l in enumerate(loss_d_list_r):
            self.log(f"Discriminator Adversarial/real {i}", l.item())

    def configure_optimizers(self):
        opt_g = optim.AdamW(
            self.generator.parameters(),
            lr=self.config.optimizer.lr,
            betas=self.config.optimizer.betas,
        )
        opt_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.optimizer.lr,
            betas=self.config.optimizer.betas,
        )
        return opt_g, opt_d
