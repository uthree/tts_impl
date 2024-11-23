import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig

from tts_impl.net.vocoder.hifigan.loss import (discriminator_loss,
                                               feature_loss, generator_loss)
from tts_impl.transforms import LogMelSpectrogram

from .discriminator import HifiganDiscriminator, HifiganDiscriminatorConfig
from .generator import HifiganGenerator, HifiganGeneratorConfig
from dataclasses import dataclass, field
from typing import List, Optional, Mapping, Any


@dataclass
class MelSpectrogramConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0


@dataclass
class OptimizerConfig:
    lr: float = 2e-4
    betas: List[int] = field(default_factory=[0.8, 0.99])


@dataclass
class HifiganLightningModuleConfig:
    generator: HifiganGeneratorConfig = field(default_factory=HifiganGeneratorConfig)
    discriminator: HifiganDiscriminatorConfig = field(default_factory=HifiganDiscriminatorConfig())
    mel: MelSpectrogramConfig = field(default_factory=MelSpectrogramConfig())
    weight_mel: float = 45.0
    weight_adv: float = 1.0
    weight_feat: float = 1.0


# HiFi-GAN from https://arxiv.org/abs/2010.05646
class HifiganLightningModule(L.LightningModule):
    def __init__(self, generator: Optional[Mapping[str, Any]] = None, discriminator: Optional[Mapping[str, Any]] = None, mel: Optional[ Mapping[str, Any]] = None, use_acoustic_features: bool = False):
        super().__init__()

        generator = generator or dict()
        discriminator = discriminator or dict()
        mel = mel or dict()

        self.automatic_optimization = False

        # flag for using data[acoustic_features] instead of mel spectrogram
        self.use_acoustic_features = use_acoustic_features 
        self.generator = HifiganGenerator(**generator)
        self.discriminator = HifiganDiscriminator(**discriminator)
        self.spectrogram = LogMelSpectrogram(**mel)

        self.save_hyperparameters()

    def training_step(self, data):
        waveform = data["waveform"]

        if self.use_acoustic_features:
            acoustic_features = data["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(waveform.sum(1)).detach()

        spec_real = self.spectrogram(waveform.sum(1)).detach()
        opt_g, opt_d = self.optimizers()

        # TODO: add loss weight configuration
        weight_mel = 45.0
        weight_feat = 1.0
        weight_adv = 1.0

        # Train G.
        fake = self.generator(acoustic_features)
        logits, fmap_fake = self.discriminator(fake)
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
        logits_fake, fmap_fake = self.discriminator(fake)
        logits_real, fmap_real = self.discriminator(waveform)
        loss_d, loss_d_list_r, loss_d_list_f = discriminator_loss(
            logits_real, logits_fake
        )

        self.toggle_optimizer(opt_d)
        self.manual_backward(loss_d)
        self.clip_gradients(opt_d, 1.0, "norm")
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # Logs
        self.log("Loss/Generator All", loss_g)
        self.log("Loss/Mel Spectrogram", loss_mel)
        self.log("Loss/Feature Matching", loss_feat)
        self.log("Loss/Generator Adversarial", loss_adv)
        self.log("Loss/Discriminator Adversarial", loss_d)

        self.log("Gen.", loss_g, prog_bar=True, logger=False)
        self.log("Dis.", loss_d, prog_bar=True, logger=False)

        for i, l in enumerate(loss_adv_list):
            self.log(f"Generator Adversarial/{i}", l)
        for i, l in enumerate(loss_d_list_f):
            self.log(f"Discriminator Adversarial/fake {i}", l)
        for i, l in enumerate(loss_d_list_r):
            self.log(f"Discriminator Adversarial/real {i}", l)

    def validation_step(self, batch):
        waveform = batch[waveform]

        if self.use_acoustic_feature:
            acoustic_features = batch["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(waveform.sum(1)).detach()

        spec_real = self.spectrogram(waveform.sum(1)).detach()
        fake = self.generator(acoustic_features)
        spec_fake = self.spectrogram(fake.sum(1))
        loss_mel = F.l1_loss(spec_fake, spec_real)

        self.log("Validation loss/Mel Spectrogram", loss_mel)

    def configure_optimizers(self):
        opt_g = optim.AdamW(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.8, 0.99),
        )
        opt_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=2e-4,
            betas=(0.8, 0.99),
        )
        return opt_g, opt_d
