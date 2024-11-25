from dataclasses import dataclass, field
from typing import Any, List, Literal, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.optim.lr_scheduler import StepLR
from tts_impl.net.base.vocoder import GanVocoderGenerator
from tts_impl.net.vocoder.hifigan import (
    HifiganDiscriminator,
    HifiganDiscriminatorConfig,
    HifiganLightningModule,
    HifiganLightningModuleConfig,
)
from tts_impl.net.vocoder.hifigan.loss import (
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from tts_impl.transforms import LogMelSpectrogram
from tts_impl.utils.config import Configuratible

from .generator import NsfhifiganGenerator, NsfhifiganGeneratorConfig


@dataclass
class NsfhifiganLightningModuleConfig(HifiganLightningModuleConfig):
    generator: NsfhifiganGeneratorConfig = field(
        default_factory=lambda: NsfhifiganGeneratorConfig
    )


class NsfhifiganLightningModule(
    HifiganLightningModule, Configuratible[NsfhifiganLightningModuleConfig]
):
    def __init__(
        self,
        generator: Optional[Mapping[str, Any]] = None,
        discriminator: Optional[Mapping[str, Any]] = None,
        mel: Optional[Mapping[str, Any]] = None,
        use_acoustic_features: bool = False,
        weight_mel: float = 45.0,
        weight_feat: float = 1.0,
        weight_adv: float = 1.0,
        lr_decay: float = 0.999,
        betas: List[float] = [0.8, 0.99],
        lr: float = 2e-4,
    ):
        super().__init__()

        generator = generator or dict()
        discriminator = discriminator or dict()
        mel = mel or dict()

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
        uv = batch.get("f0", None)

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
        uv = batch.get("f0", None)

        if self.use_acoustic_features:
            acoustic_features = batch["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(waveform.sum(1)).detach()

        spec_real = self.spectrogram(waveform.sum(1)).detach()
        fake = self.generator(acoustic_features, f0=f0, uv=uv)
        spec_fake = self.spectrogram(fake.sum(1))
        loss_mel = F.l1_loss(spec_fake, spec_real)
        self.log("validation loss/mel spectrogram", loss_mel)

        return loss_mel
