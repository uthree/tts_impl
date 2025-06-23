from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from transformers import HubertModel

from tts_impl.net.vocoder.ddsp import SubtractiveVocoder
from tts_impl.net.vocoder.hifigan import HifiganDiscriminator
from tts_impl.net.vocoder.hifigan.loss import (discriminator_loss,
                                               feature_loss, generator_loss)
from tts_impl.transforms import LogMelSpectrogram
from tts_impl.utils.config import derive_config

from .generator import Decoder, Encoder

discriminator_cfg_default = HifiganDiscriminator.Config()
discriminator_cfg_default.msd.scales = [1]
discriminator_cfg_default.mpd.periods = [2, 3, 5, 7, 11]
discriminator_cfg_default.mpd.channels_max = 256
discriminator_cfg_default.mpd.channels_mul = 2
discriminator_cfg_default.mrsd.n_fft = [512, 1024, 2048]
discriminator_cfg_default.mrsd.hop_size = [50, 120, 240]
discriminator_cfg_default.mrxd.n_fft = [512, 1024, 2048]
discriminator_cfg_default.mrxd.hop_size = [50, 120, 240]


def slice_segments(x, ids_str, segment_size=8192):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=8192):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


# normalize tensor for tensorboard's image logging
def normalize(x: torch.Tensor):
    x = x.to(torch.float)
    mu = x.mean()
    x = x - mu
    x = x / torch.clamp_min(x.abs().max(), min=1e-8)
    return x


@derive_config
class DdspVoiceConversionLightningModule(LightningModule):
    def __init__(
        self,
        encoder: Encoder.Config = Encoder.Config(),
        decoder: Decoder.Config = Decoder.Config(),
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
        discriminator: HifiganDiscriminator.Config = discriminator_cfg_default,
        mel: LogMelSpectrogram.Config = LogMelSpectrogram.Config(),
        n_speakers: int = 1024,
        gin_channels: int = 256,
        use_acoustic_features: bool = False,
        weight_mel: float = 45.0,
        weight_feat: float = 1.0,
        weight_adv: float = 1.0,
        lr_decay: float = 0.999,
        betas: List[float] = [0.8, 0.99],
        segment_size: int = 8192,
        lr: float = 2e-4,
        ssl_model_name: str = "rinna/japanese-hubert-base",
        ssl_model_sample_rate: int = 16000,
    ):
        super().__init__()
        self.automatic_optimization = False

        # flag for using data[acoustic_features] instead of mel spectrogram
        self.use_acoustic_features = use_acoustic_features
        self.encoder = Encoder(**encoder)
        self.decoder = Decoder(**decoder)
        self.vocoder = SubtractiveVocoder(**vocoder)
        self.discriminator = HifiganDiscriminator(**discriminator)
        self.spectrogram = LogMelSpectrogram(**mel)
        self.gin_channels = gin_channels
        self.n_speakers = n_speakers
        self.weight_mel = weight_mel
        self.weight_adv = weight_adv
        self.weight_feat = weight_feat
        self.lr_decay = lr_decay
        self.lr = lr
        self.betas = betas
        self.segment_size = segment_size
        self.ssl_model_name = ssl_model_name
        self.ssl_model_sample_rate = ssl_model_sample_rate
        self.speaker_embedding = nn.Embedding(n_speakers, gin_channels)

        # initialize SSL model
        ssl_model = HubertModel.from_pretrained(ssl_model_name)
        ssl_model.eval()
        self.ssl_model = ssl_model
        self.ssl_d_model = self.ssl_model.config.hidden_size
        self.emb2ssl = nn.Linear(
            self.encoder.phoneme_embedding_dim, self.ssl_d_model, False
        )

        self.save_hyperparameters()

    def training_step(self, batch: dict):
        real = batch["waveform"]
        f0 = batch["f0"]
        sid = batch.get("speaker_id", None)

        if "acoustic_features" in batch:
            acoustic_features = batch["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(real.sum(1)).detach()

        fake, slice_ids = self._adversarial_training_step(
            acoustic_features, real, f0, sid
        )
        self._discriminator_training_step(real, fake, slice_ids)

    def _adversarial_training_step(self, af, real, f0, sid):
        # real spec.
        spec_real = self.spectrogram(real).detach()

        # generator forward
        g = self.speaker_embedding(sid).unsqueeze(2)
        z, f0_logits, _h_last = self.encoder(af)
        z = z.detach()
        loss_f0, loss_uv = self.encoder.f0_loss(f0_logits, f0)

        # SSL
        with torch.no_grad():
            wf_resampled = torchaudio.functional.resample(
                real, self.vocoder.sample_rate, self.ssl_model_sample_rate
            )
            ssl_feats = self.ssl_model.forward(
                wf_resampled.sum(dim=1), output_hidden_states=True
            ).hidden_states[9]
            ssl_feats = F.interpolate(
                ssl_feats.transpose(1, 2), size=z.shape[2], mode="linear"
            ).transpose(1, 2)
            # instance normalization
            ssl_feats = (ssl_feats - ssl_feats.mean(dim=1, keepdim=True)) / torch.clamp_min(ssl_feats.std(dim=1, keepdim=True, 1e-5))

        z_proj = self.emb2ssl(z.transpose(1, 2))
        loss_ssl = F.huber_loss(z_proj, ssl_feats)
        se, ap, _h_last = self.decoder(z, h_0=None, g=g)
        fake = self.vocoder.synthesize(f0, se, ap).unsqueeze(1)

        # spectrogram
        spec_fake = self.spectrogram(fake)

        # get optimizer
        opt_g, _opt_d = self.optimizers()

        # slice waveforms
        real_slice, slice_ids = rand_slice_segments(
            real, segment_size=self.segment_size
        )
        fake_slice = slice_segments(fake, slice_ids, segment_size=self.segment_size)

        # forward pass
        logits, fmap_fake = self.discriminator(fake_slice)
        _, fmap_real = self.discriminator(real_slice)
        loss_adv, loss_adv_list = generator_loss(logits)
        loss_feat = feature_loss(fmap_real, fmap_fake)
        loss_mel = F.l1_loss(spec_fake, spec_real)
        loss_g = (
            loss_mel * self.weight_mel
            + loss_feat * self.weight_feat
            + loss_adv * self.weight_adv
            + loss_uv
            + loss_f0
            + loss_ssl
        )

        # update parameters
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad(set_to_none=True)
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
        self.log("train loss/ssl distillation", loss_ssl)
        self.log("train loss/pitch estimation", loss_f0)
        self.log("train loss/vuv", loss_uv)
        self.log("G", loss_g, prog_bar=True, logger=False)

        # return fake, slice ids
        return fake, slice_ids

    def _test_or_validate_batch(self, batch, bid):
        waveform = batch["waveform"]
        f0 = batch.get("f0", None)
        uv = batch.get("uv", None)
        sid = batch.get("speaker_id", None)

        if self.use_acoustic_features:
            acoustic_features = batch["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(waveform.sum(1)).detach()

        spec_real = self.spectrogram(waveform).detach()
        g = self.speaker_embedding(sid).unsqueeze(2)
        z, f0_logits, _ = self.encoder(acoustic_features)
        f0_hat = self.encoder.decode_f0(f0_logits)
        se, ap, _ = self.decoder(z, g=g)
        recon = self.vocoder.synthesize(f0_hat, se, ap).unsqueeze(1)

        spec_recon = self.spectrogram(recon)
        loss_mel = F.l1_loss(spec_recon, spec_real)
        self.log("validation loss/mel spectrogram", loss_mel)

        for i in range(recon.shape[0]):
            f = recon[i].sum(dim=0, keepdim=True).detach().cpu()
            r = waveform[i].sum(dim=0, keepdim=True).detach().cpu()
            spec_recon_img = normalize(spec_recon[i, 0].detach().cpu().flip(0))
            spec_real_img = normalize(spec_real[i, 0].detach().cpu().flip(0))
            self.logger.experiment.add_audio(
                f"reconstructed waveform/{bid}_{i}",
                f,
                self.current_epoch,
                sample_rate=self.vocoder.sample_rate,
            )
            self.logger.experiment.add_audio(
                f"reference waveform/{bid}_{i}",
                r,
                self.current_epoch,
                sample_rate=self.vocoder.sample_rate,
            )
            self.logger.experiment.add_image(
                f"reconstructed mel spectrogram/{bid}_{i}",
                spec_recon_img,
                self.current_epoch,
                dataformats="HW",
            )
            self.logger.experiment.add_image(
                f"reference mel spectrogram/{bid}_{i}",
                spec_real_img,
                self.current_epoch,
                dataformats="HW",
            )

        return loss_mel

    def _discriminator_training_step(
        self, real: torch.Tensor, fake: torch.Tensor, slice_ids
    ):
        opt_g, opt_d = self.optimizers()  # get optimizer

        # slice waveform
        real_slice = slice_segments(real.detach(), slice_ids, self.segment_size)
        fake_slice = slice_segments(fake.detach(), slice_ids, self.segment_size)

        # forward pass
        logits_fake, _ = self.discriminator(fake_slice)
        logits_real, _ = self.discriminator(real_slice)
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
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()
        self.log("scheduler/learning rate", sch_g.get_last_lr()[0])

    def validation_step(self, batch, id):
        return self._test_or_validate_batch(batch, id)

    def test_step(self, batch, id):
        return self._test_or_validate_batch(batch, id)

    def configure_optimizers(self):
        opt_g = optim.AdamW(
            nn.ModuleList([self.encoder, self.decoder]).parameters(),
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
