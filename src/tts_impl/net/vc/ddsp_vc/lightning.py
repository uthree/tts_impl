from dataclasses import dataclass, field
from typing import Any, List, Literal, Mapping, Optional, Tuple

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

from .generator import DdspvcGenerator
from .speaker_classifier import SpeakerClassifier


def crop_center(waveform: torch.Tensor, length: int = 8192):
    l = (waveform.shape[2] // 2) - length // 2
    r = (waveform.shape[2] // 2) + length // 2
    return waveform[:, :, l:r]


# normalize tensor for tensorboard's image logging
def normalize(x: torch.Tensor):
    x = x.to(torch.float)
    mu = x.mean()
    x = x - mu
    x = x / torch.clamp_min(x.abs().max(), min=1e-8)
    return x

_melspec_default = LogMelSpectrogram.Config()
_melspec_default.sample_rate = 24000


@derive_config
class DdspVcLightningModule(LightningModule):
    def __init__(
        self,
        generator: DdspvcGenerator.Config = DdspvcGenerator.Config(),
        discriminator: HifiganDiscriminator.Config = HifiganDiscriminator.Config(),
        speaker_classifier: SpeakerClassifier.Config = SpeakerClassifier.Config(),
        mel: LogMelSpectrogram.Config = _melspec_default,
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
        self.generator = DdspvcGenerator(**generator)
        self.discriminator = HifiganDiscriminator(**discriminator)
        self.spectrogram = LogMelSpectrogram(**mel)
        self.speaker_classifier = SpeakerClassifier(**speaker_classifier)
        self.weight_mel = weight_mel
        self.weight_adv = weight_adv
        self.weight_feat = weight_feat
        self.lr_decay = lr_decay
        self.lr = lr
        self.betas = betas

        self.save_hyperparameters()

    def training_step(self, batch: dict):
        real = batch["waveform"]
        sid = batch["speaker_id"]
        f0 = batch["f0"]

        if "acoustic_features" in batch:
            acoustic_features = batch["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(real.sum(1)).detach()

        fake, z = self._generator_training_step(real, acoustic_features, f0, sid)
        self._discriminator_training_step(crop_center(real), crop_center(fake))
        self._speaker_classifier_training_step(z, sid)

    def _generator_training_step(self, real: torch.Tensor, af: torch.Tensor, f0: torch.Tensor, sid: torch.Tensor):
        opt_g, opt_d, opt_c = self.optimizers()
        opt_g.zero_grad()
        self.toggle_optimizer(opt_g)
        fake, z, loss_f0, loss_uv = self.generator.forward(af, f0, sid)
        spk_logits = self.speaker_classifier(z)
        loss_spk_grl = F.cross_entropy(spk_logits, sid)
        loss_g_adv = self._adversarial_loss(crop_center(real), crop_center(fake))
        loss_g = - loss_spk_grl + loss_g_adv + loss_f0 + loss_uv
        self.manual_backward(loss_g)
        opt_g.step()

        self.log(f"train loss/generator GRL", loss_spk_grl)
        self.log(f"train loss/pitch estimation", loss_f0)
        self.log(f"train loss/uv estimation", loss_uv)
        self.untoggle_optimizer(opt_g)
        return fake, z.detach()

    def _speaker_classifier_training_step(self, z: torch.Tensor, sid: torch.Tensor):
        z = z.detach()
        opt_g, opt_d, opt_c = self.optimizers() # get optimizer
        self.toggle_optimizer(opt_c)
        opt_c.zero_grad()
        spk_logits = self.speaker_classifier(z)
        loss_spk = F.cross_entropy(spk_logits, sid)
        self.manual_backward(loss_spk)
        self.clip_gradients(opt_c, 1.0, "norm")
        opt_c.step()
        self.untoggle_optimizer(opt_c)

    def _adversarial_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        # spectrogram
        spec_real = self.spectrogram(real).detach()
        spec_fake = self.spectrogram(fake)

        # get optimizer
        opt_g, opt_d, opt_c = self.optimizers()

        # forward pass
        logits, fmap_fake = self.discriminator(fake)
        _, fmap_real = self.discriminator(real)
        loss_adv, loss_adv_list = generator_loss(logits)
        loss_feat = feature_loss(fmap_real, fmap_fake)
        loss_mel = F.l1_loss(spec_fake, spec_real)
        loss_g_adv = (
            loss_mel * self.weight_mel
            + loss_feat * self.weight_feat
            + loss_adv * self.weight_adv
        )

        # logs
        for i, l in enumerate(loss_adv_list):
            self.log(f"generator adversarial/{i}", l)
        self.log("train loss/mel spectrogram", loss_mel)
        self.log("train loss/feature matching", loss_feat)
        self.log("train loss/generator adversarial", loss_adv)
        return loss_g_adv

    def _test_or_validate_batch(self, batch, bid):
        waveform = batch["waveform"]
        sid = batch["speaker_id"]
        f0 = batch.get("f0", None)
        uv = batch.get("uv", None)

        if self.use_acoustic_features:
            acoustic_features = batch["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(waveform.sum(1)).detach()

        spec_real = self.spectrogram(waveform.sum(1)).detach()
        fake, z, loss_f0, loss_uv= self.generator.forward(acoustic_features, f0=f0, sid=sid)
        spec_fake = self.spectrogram(fake.sum(1))
        converted, _, _, _= self.generator.forward(acoustic_features, f0=f0, sid=torch.roll(sid, 1, dims=(0,)))
        spec_converted = self.spectrogram(converted.sum(1))
        loss_mel = F.l1_loss(spec_fake, spec_real)
        self.log("validation loss/mel spectrogram", loss_mel)

        for i in range(fake.shape[0]):
            f = fake[i].sum(dim=0, keepdim=True).detach().cpu()
            r = waveform[i].sum(dim=0, keepdim=True).detach().cpu()
            c = converted[i].sum(dim=0, keepdim=True).detach().cpu()
            spec_real_img = normalize(spec_real[i])
            spec_fake_img = normalize(spec_fake[i])
            spec_converted_img = normalize(spec_converted[i])
            self.logger.experiment.add_audio(
                f"reconstructed waveform/{bid}_{i}",
                f,
                self.current_epoch,
                sample_rate=self.generator.sample_rate,
            )
            self.logger.experiment.add_audio(
                f"reference waveform/{bid}_{i}",
                r,
                self.current_epoch,
                sample_rate=self.generator.sample_rate,
            )
            self.logger.experiment.add_image(
                f"synthesized mel spectrogram/{bid}_{i}",
                spec_fake_img.flip(0, ),
                self.current_epoch,
                dataformats="HW",
            )
            self.logger.experiment.add_image(
                f"reference mel spectrogram/{bid}_{i}",
                spec_real_img.flip(0,),
                self.current_epoch,
                dataformats="HW",
            )
            src_sid = sid[i]
            tgt_sid = torch.roll(sid, 1, dims=(0,))[i]
            self.logger.experiment.add_image(
                f"converted mel spectrogram/{bid}_{i}_{src_sid}_to_{tgt_sid}",
                spec_converted_img.flip(0,),
                self.current_epoch,
                dataformats="HW",
            )
            self.logger.experiment.add_audio(
                f"converted waveform/{bid}_{i}_{src_sid}_to_{tgt_sid}",
                c,
                self.current_epoch,
                sample_rate=self.generator.sample_rate,
            )

        return loss_mel

    def _discriminator_training_step(self, real: torch.Tensor, fake: torch.Tensor):
        opt_g, opt_d, opt_c = self.optimizers()  # get optimizer

        # forward pass
        fake = fake.detach()  # stop gradient
        logits_fake, _ = self.discriminator(fake)
        logits_real, _ = self.discriminator(real)
        loss_d, loss_d_list_r, loss_d_list_f = discriminator_loss(
            logits_real, logits_fake
        )

        # update parameters
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad(set_to_none=True)
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
        sch_g, sch_d, sch_c = self.lr_schedulers()
        sch_g.step()
        sch_d.step()
        sch_c.step()
        self.log("scheduler/learning rate", sch_g.get_last_lr()[0])

    @torch.no_grad
    def validation_step(self, batch, id):
        return self._test_or_validate_batch(batch, id)

    @torch.no_grad
    def test_step(self, batch, id):
        return self._test_or_validate_batch(batch, id)

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
        opt_c = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=self.betas,
        )
        sch_c = StepLR(opt_g, 1, self.lr_decay)
        return [opt_g, opt_d, opt_c], [sch_g, sch_d, sch_c]
