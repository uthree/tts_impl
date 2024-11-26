from dataclasses import dataclass, field
from typing import Any, List, Literal, Mapping, Optional

import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from tts_impl.net.vocoder.hifigan import HifiganDiscriminator, HifiganLightningModule
from tts_impl.transforms import LogMelSpectrogram
from tts_impl.utils.config import derive_config

from .generator import NsfhifiganGenerator


@derive_config
class NsfhifiganLightningModule(HifiganLightningModule):
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
        LightningModule.__init__(self)
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
