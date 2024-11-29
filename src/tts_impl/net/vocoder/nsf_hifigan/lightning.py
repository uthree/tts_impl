from dataclasses import dataclass, field
from typing import Any, List, Literal, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from torch.optim.lr_scheduler import StepLR
from tts_impl.net.vocoder.hifigan import HifiganDiscriminator
from tts_impl.net.vocoder.hifigan.loss import (
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from tts_impl.transforms import LogMelSpectrogram
from tts_impl.utils.config import derive_config

from .generator import NsfhifiganGenerator


@derive_config
class NsfhifiganLightningModule(LightningModule):
    def __init__(
        self,
        generator: NsfhifiganGenerator.Config = NsfhifiganGenerator.Config(),
        discriminator: HifiganDiscriminator.Config = HifiganDiscriminator.Config(),
        mel: LogMelSpectrogram.Config = LogMelSpectrogram.Config(),
        use_acoustic_features: bool = False,
        weight_mel: float = 45.0,
        weight_feat: float = 1.0,
        weight_adv: float = 1.0,
        lr_decay: float = 0.999,
        betas: List[float] = [0.8, 0.99],
        lr: float = 2e-4,
    ):
        super().__init__()
        self.automatic_optimization = False

        # flag for using data[acoustic_features] instead of mel spectrogram
        self.use_acoustic_features = use_acoustic_features
        self.generator = NsfhifiganGenerator(**generator)
        self.discriminator = HifiganDiscriminator(**discriminator)
        self.spectrogram = LogMelSpectrogram(**mel)
        self.weight_mel = weight_mel
        self.weight_adv = weight_adv
        self.weight_feat = weight_feat
        self.lr_decay = lr_decay
        self.lr = lr
        self.betas = betas

        self.save_hyperparameters()

    def training_step(self, batch: dict):
        real = batch["waveform"]
        f0 = batch.get("f0", None)
        uv = batch.get("uv", None)

        if "acoustic_features" in batch:
            acoustic_features = batch["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(real.sum(1)).detach()

        fake = self.generator(acoustic_features, f0=f0, uv=uv)
        self.generator_training_step(real, fake)
        self.discriminator_training_step(real, fake)

    def generator_training_step(self, real: torch.Tensor, fake: torch.Tensor):
        # spectrogram
        spec_real = self.spectrogram(real).detach()
        spec_fake = self.spectrogram(fake)

        # get optimizer
        opt_g, opt_d = self.optimizers()

        # forward pass
        logits, fmap_fake = self.discriminator(fake)
        _, fmap_real = self.discriminator(real)
        loss_adv, loss_adv_list = generator_loss(logits)
        loss_feat = feature_loss(fmap_real, fmap_fake)
        loss_mel = F.l1_loss(spec_fake, spec_real)
        loss_g = (
            loss_mel * self.weight_mel
            + loss_feat * self.weight_feat
            + loss_adv * self.weight_adv
        )

        # update parameters
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        self.clip_gradients(opt_g, 1.0, "norm")
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # logs
        for i, l in enumerate(loss_adv_list):
            self.log(f"generator adversarial/{i}", l)
        self.log("train loss/generator total", loss_g)
        self.log("train loss/mel spectrogram", loss_mel)
        self.log("train loss/feature matching", loss_feat)
        self.log("train loss/generator adversarial", loss_adv)
        self.log("G", loss_g, prog_bar=True, logger=False)

    def _test_or_validate_batch(self, batch):
        waveform = batch["waveform"]
        f0 = batch.get("f0", None)
        uv = batch.get("uv", None)

        if self.use_acoustic_features:
            acoustic_features = batch["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(waveform.sum(1)).detach()

        spec_real = self.spectrogram(waveform).detach()
        fake = self.generator(acoustic_features, f0=f0, uv=uv)
        spec_fake = self.spectrogram(fake)
        loss_mel = F.l1_loss(spec_fake, spec_real)
        self.log("validation loss/mel spectrogram", loss_mel)
        return loss_mel

    def discriminator_training_step(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        opt_g, opt_d = self.optimizers()  # get optimizer

        # forward pass
        fake = fake.detach()  # stop gradient
        logits_fake, _ = self.discriminator(fake)
        logits_real, _ = self.discriminator(real)
        loss_d, loss_d_list_r, loss_d_list_f = discriminator_loss(
            logits_real, logits_fake
        )

        # update parameters
        self.toggle_optimizer(opt_d)
        self.manual_backward(loss_d)
        self.clip_gradients(opt_d, 1.0, "norm")
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # logs
        for i, l in enumerate(loss_d_list_f):
            self.log(f"discriminator adversarial/fake {i}", l)
        for i, l in enumerate(loss_d_list_r):
            self.log(f"discriminator adversarial/real {i}", l)
        self.log("train loss/discriminator", loss_d)
        self.log("D", loss_d, prog_bar=True, logger=False)

    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()
        self.log("scheduler/learning Rate", sch_g.get_last_lr()[0])

    def validation_step(self, batch):
        return self._test_or_validate_batch(batch)

    def test_step(self, batch):
        return self._test_or_validate_batch(batch)

    def configure_optimizers(self):
        opt_g = optim.AdamW(
            self.generator.parameters(),
            lr=self.lr,
            betas=self.betas,
        )
        sch_g = StepLR(opt_g, 1, self.lr_decay)
        opt_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=self.betas,
        )
        sch_d = StepLR(opt_g, 1, self.lr_decay)
        return [opt_g, opt_d], [sch_g, sch_d]
