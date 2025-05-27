import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tts_impl.net.tts.vits.commons import slice_segments
from tts_impl.net.tts.vits.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from tts_impl.net.vocoder.hifigan import HifiganDiscriminator
from tts_impl.net.vocoder.loss import MultiResolutionSTFTLoss
from tts_impl.transforms import LogMelSpectrogram
from tts_impl.utils.config import derive_config

from .models import NsfvitsGenerator

_vits_discriminator_config = HifiganDiscriminator.Config()
_vits_discriminator_config.msd.scales = [1]
_vits_discriminator_config.mpd.periods = [2, 3, 5, 7, 11]
_vits_discriminator_config.mrsd.n_fft = [240, 400, 600]
_vits_discriminator_config.mrsd.hop_size = [50, 100, 200]


# normalize tensor for tensorboard's image logging
def normalize(x: torch.Tensor):
    x = x.to(torch.float)
    mu = x.mean()
    x = x - mu
    x = x / torch.clamp_min(x.abs().max(), min=1e-8)
    return x


@derive_config
class NsfvitsLightningModule(L.LightningModule):
    def __init__(
        self,
        generator: NsfvitsGenerator.Config = NsfvitsGenerator.Config(),
        discriminator: HifiganDiscriminator.Config = _vits_discriminator_config,
        mel: LogMelSpectrogram.Config = LogMelSpectrogram.Config(),
        mr_stft_loss: MultiResolutionSTFTLoss.Config = MultiResolutionSTFTLoss.Config(),
        weight_mel: float = 25.0,
        weight_stft: float = 20.0,
        weight_feat: float = 1.0,
        weight_adv: float = 1.0,
        lr: float = 2e-4,
        lr_decay: float = 0.9998749453,
        betas: list[float] = [0.8, 0.99],
    ):
        super().__init__()
        self.automatic_optimization = False

        self.generator = NsfvitsGenerator(**generator)
        self.discriminator = HifiganDiscriminator(**discriminator)
        self.spectrogram = LogMelSpectrogram(**mel)
        self.mr_stft_loss = MultiResolutionSTFTLoss(**mr_stft_loss)

        self.weight_stft = weight_stft
        self.weight_mel = weight_mel
        self.weight_adv = weight_adv
        self.weight_feat = weight_feat

        self.lr_decay = lr_decay
        self.lr = lr
        self.betas = betas

        self.save_hyperparameters()

    def training_step(self, batch):
        # expand batch
        waveform = batch["waveform"]
        if "acoustic_features" in batch:
            y = batch["acoustic_features"]
        else:
            y = self.spectrogram(waveform.sum(1)).detach()
        y_lengths = batch["acoustic_features_lengths"]
        x = batch["phonemes"]
        x_lengths = batch["phonemes_lengths"]
        sid = batch.get("speaker_id", None)
        w = batch.get("duration", None)
        f0 = batch.get("f0", None)

        # generator step
        real, fake = self._generator_training_step(
            x, x_lengths, y, y_lengths, waveform, f0, sid=sid, w=w
        )

        # discriminator step
        self._discriminator_training_step(real, fake)

    def _generator_training_step(
        self, x, x_lengths, y, y_lengths, waveform, f0, sid=None, w=None
    ):
        opt_g, _opt_d = self.optimizers()  # get optimizer

        # forward pass
        real, fake, loss_gen_tts = self._generator_forward(
            x, x_lengths, y, y_lengths, waveform, f0, sid=sid, w=w
        )
        loss_gen_vocoder = self._vocoder_adversarial_loss(real, fake)
        loss_g = loss_gen_tts + loss_gen_vocoder

        # logs
        self.log("train loss/generator total", loss_g)
        self.log("G", loss_g, prog_bar=True, logger=False)

        # take generator's gradient descent
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(loss_g)
        self.clip_gradients(opt_g, 1.0, "norm")
        opt_g.step()
        self.untoggle_optimizer(opt_g)
        fake = fake.detach()
        real = real.detach()
        return real, fake

    def _generator_forward(
        self, x, x_lengths, y, y_lengths, waveform, f0, sid=None, w=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # get frame size and segment size
        segment_size = self.generator.segment_size
        dec_frame_size = self.generator.dec.frame_size

        outputs = self.generator.forward(
            x, x_lengths, y, y_lengths, f0, sid=sid, w=w
        )  # forward pass

        # expand return dict.
        z_p = outputs["z_p"]
        logs_q = outputs["logs_q"]
        m_p = outputs["m_p"]
        logs_p = outputs["logs_p"]
        z_mask = outputs["y_mask"]
        ids_slice = outputs["ids_slice"]
        fake = outputs["fake"]

        # losses
        loss_dur = outputs["loss_dur"]
        loss_dur = loss_dur.mean()
        loss_f0 = outputs["loss_f0"]
        loss_uv = outputs["loss_uv"]
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)

        # logs
        self.log("train loss/KL divergence", loss_kl)
        self.log("train loss/duration", loss_dur)
        self.log("train loss/pitch estimation", loss_f0)
        self.log("train loss/uv estimation", loss_uv)

        # slice real input
        real = slice_segments(
            waveform, ids_slice * dec_frame_size, segment_size * dec_frame_size
        ).detach()

        loss = loss_dur + loss_kl + loss_f0 + loss_uv
        return real, fake, loss

    def _vocoder_adversarial_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        # spectrogram
        spec_real = self.spectrogram(real).detach()
        spec_fake = self.spectrogram(fake)

        # forward pass
        logits, fmap_fake = self.discriminator(fake)
        _, fmap_real = self.discriminator(real)
        loss_adv, loss_adv_list = generator_loss(logits)
        loss_feat = feature_loss(fmap_real, fmap_fake)
        loss_mel = F.l1_loss(spec_fake, spec_real)
        loss_stft_sc, loss_stft_mag = self.mr_stft_loss(
            fake.sum(dim=1), real.sum(dim=1)
        )
        loss_stft = loss_stft_sc + loss_stft_mag
        loss_g = (
            loss_stft * self.weight_stft
            + loss_mel * self.weight_mel
            + loss_feat * self.weight_feat
            + loss_adv * self.weight_adv
        )

        # logs
        for i, l in enumerate(loss_adv_list):
            self.log(f"generator adversarial/{i}", l)
        self.log("train loss/spectral convergence", loss_stft_sc)
        self.log("train loss/spectral magnitude", loss_stft_mag)
        self.log("train loss/mel spectrogram", loss_mel)
        self.log("train loss/feature matching", loss_feat)
        self.log("train loss/generator adversarial", loss_adv)
        return loss_g

    def _discriminator_training_step(self, real: torch.Tensor, fake: torch.Tensor):
        _opt_g, opt_d = self.optimizers()  # get optimizer

        # forward pass
        fake = fake.detach()  # stop gradient
        real = real.detach()
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

    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()
        self.log("scheduler/learning rate", sch_g.get_last_lr()[0])

    @torch.no_grad
    def validation_step(self, batch, bid):
        real_waveform = batch["waveform"]
        spec_real = self.spectrogram(real_waveform).detach()
        max_len = spec_real.shape[3]
        fake_waveform = self.generator.infer(
            batch["phonemes"],
            batch["phonemes_lengths"],
            batch["speaker_id"],
            max_len=max_len,
        )
        spec_fake = self.spectrogram(fake_waveform)

        for i in range(fake_waveform.shape[0]):
            r = real_waveform[i].sum(dim=0, keepdim=True).detach().cpu()
            f = fake_waveform[i].sum(dim=0, keepdim=True).detach().cpu()
            spec_fake_img = normalize(spec_fake[i, 0].detach().cpu().flip(0))
            spec_real_img = normalize(spec_real[i, 0].detach().cpu().flip(0))
            self.logger.experiment.add_audio(
                f"synthesized waveform/{bid}_{i}",
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
                spec_fake_img,
                self.current_epoch,
                dataformats="HW",
            )
            self.logger.experiment.add_image(
                f"reference mel spectrogram/{bid}_{i}",
                spec_real_img,
                self.current_epoch,
                dataformats="HW",
            )
