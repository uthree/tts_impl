# https://arxiv.org/abs/2401.10460
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from torch.optim.lr_scheduler import StepLR
from torchaudio.functional import melscale_fbanks
from torchaudio.models import Emformer

from tts_impl.functional.ddsp import impulse_train
from tts_impl.net.base.vocoder import GanVocoderDiscriminator, GanVocoderGenerator
from tts_impl.net.common.stft import STFT
from tts_impl.net.vocoder.discriminator import MultiResolutionStftDiscriminator
from tts_impl.transforms import LogMelSpectrogram
from tts_impl.utils.config import derive_config


@derive_config
class UltraLighweightDdsp(nn.Module):
    """
    Unofficial implementation of Meta AI's Ultra-Lightweight DDSP Vocoder
    based https://arxiv.org/abs/2401.10460
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = 128,
        n_fft: int = 512,
        n_mels: int = 12,
        fmin: float = 0,
        fmax: float = 8000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fft_bin = n_fft // 2 + 1
        self.fmin = fmin
        self.fmax = fmax

        mel_fbank = melscale_fbanks(
            self.fft_bin, fmin, fmax, n_mels, self.sample_rate
        )  # [n_freqs, n_mels]
        self.register_buffer(
            "mel_fbank", mel_fbank
        )  # register as non-trainable parameter
        self.stft = STFT(n_fft, hop_length, n_fft)

    def forward(
        self, f0: torch.Tensor, p: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f0: shape=(batch_size, L)
            p: shape=(batch_size, n_mels, L)
            v: shape=(fft_bin, n_mels, L)

            where, fft_bin = n_fft // 2 + 1.

        Returns:
            x_hat: shape=(batch_size, L*hop_length)
        """

        # Osscilate impulse
        e_imp = impulse_train(
            f0, self.hop_length, self.sample_rate
        )  # [B, L * hop_length]
        # Energy normaliziaton
        m = torch.rsqrt(
            F.interpolate(f0.unsqueeze(1), scale_factor=self.hop_length).squeeze(1)
        )
        e_imp = e_imp * m

        # Oscillate noise
        e_noise = torch.randn_like(e_imp)

        # STFT
        e_imp_r, e_imp_i = self.stft.transform(e_imp)
        e_noise_r, e_noise_i = self.stft.transform(e_noise)

        # expand p
        p = torch.matmul(self.mel_fbank, p)
        p = F.pad(p, (1, 0), mode="replicate")

        # source signal
        s_r = p * e_imp_r + (1 - p) * e_noise_r
        s_i = p * e_imp_i + (1 - p) * e_noise_i

        # vocal tract filter
        v = F.pad(v, (1, 0), mode="replicate")
        s_r = s_r * v
        s_i = s_i * v

        # iSTFT
        x_hat = self.stft.inverse(s_r, s_i)
        return x_hat


@derive_config
class AcousticModel(nn.Module):
    def __init__(
        self,
        in_dim: int = 80,
        out_dim: int = 270,
        num_heads: int = 2,
        inter_dim: int = 128,
        ffn_dim: int = 512,
        seg_size: int = 32,
        memory: int = 4,
        left_context: int = 12,
        right_context: int = 12,
        post_inter_dim: int = 199,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.right_context = right_context
        self.inter_dim = inter_dim
        self.prenet = nn.Sequential(
            nn.Linear(in_dim, inter_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.emformer = Emformer(
            inter_dim,
            num_heads,
            ffn_dim,
            num_layers,
            seg_size,
            dropout,
            "silu",
            left_context,
            right_context,
            memory,
        )
        self.postnet = nn.Sequential(
            nn.Linear(inter_dim, post_inter_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(post_inter_dim, out_dim),
        )

    def forward(self, x, x_lengths) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, channels, length)
            x_lengths: shape=(batch_size), int
        Returns:
            x: shape=(batch_size, channels, length)
        """
        x = x.transpose(1, 2)
        x = self.prenet(x)
        right_pad = torch.zeros(
            x.shape[0], self.right_context, self.inter_dim, device=x.device
        )
        x = torch.cat([x, right_pad], dim=1)
        x, _ = self.emformer(x, x_lengths)
        x = self.postnet(x)
        x = x.transpose(1, 2)
        return x


# TODO: DiscriminatorはとりあえずMRDにしておく、あとで専用のものにする。
# TODO: とりあえずHiFi_GANのものにしているので、あとで論文通りの損失関数にする
# TODO: Fix nan
@derive_config
class UltraLightweightVocoderGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        acoustic_model: AcousticModel.Config = AcousticModel.Config(),
        ddsp: UltraLighweightDdsp.Config = UltraLighweightDdsp.Config(),
    ):
        super().__init__()
        self.acoustic_model = AcousticModel(**acoustic_model)
        self.ddsp = UltraLighweightDdsp(**ddsp)
        self.n_mels = self.ddsp.n_mels
        self.fft_bin = self.ddsp.fft_bin

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, in_dim, length)
            f0: shape=(batch_size, length)
            x_lengths: shape=(batch_size), int

        Returns:
            wf: shape=(batch_size, 1, length * frame_size)
            f0: shape=(batch_size, length)
            p: shape=(batch_size, n_mels, length)
            v: shape=(batch_size, fft_bin, length)
        """
        if x_lengths is None:
            x_lengths = torch.full((x.shape[0],), x.shape[2], device=x.device)
        x = self.acoustic_model.forward(x, x_lengths)
        log_f0_hat, p, v = torch.split(x, (1, self.n_mels, self.fft_bin), dim=1)
        p = torch.sigmoid(p)
        v = torch.exp(v)
        f0_hat = torch.exp(log_f0_hat)
        wf = self.ddsp.forward(f0, p, v)
        f0_hat = f0_hat.squeeze(1)
        return wf, f0_hat, p, v

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, in_dim, length)
            x_lengths: shape=(batch_size), int

        Returns:
            wf: shape=(batch_size, 1, length * frame_size)
        """
        if x_lengths is None:
            x_lengths = torch.full((x.shape[0],), x.shape[2], device=x.device)
        x = self.acoustic_model.forward(x, x_lengths)
        log_f0, p, v = torch.split(x, (1, self.n_mels, self.fft_bin), dim=1)
        p = torch.sigmoid(p)
        v = torch.exp(v)
        f0 = torch.exp(log_f0)
        f0 = f0.squeeze(1)
        wf = self.ddsp.forward(f0, p, v)
        return wf


@derive_config
class UltraLightweightVocoderDiscriminator(MultiResolutionStftDiscriminator):
    pass


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            l = torch.mean(torch.abs(rl - gl))
            loss += l
    return loss * 2


def safe_log(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def f0_loss(f0, f0_hat):
    return F.mse_loss(safe_log(f0), safe_log(f0_hat))


_default_mel_cfg = LogMelSpectrogram.Config()
_default_mel_cfg.sample_rate = 24000
_default_mel_cfg.hop_length = 128
_default_mel_cfg.n_fft = 512
_default_mel_cfg.n_mels = 80


@derive_config
class UltraLightweightVocoderLightningModule(LightningModule):
    def __init__(
        self,
        generator: UltraLightweightVocoderGenerator.Config = UltraLightweightVocoderGenerator.Config(),
        discriminator: UltraLightweightVocoderDiscriminator.Config = UltraLightweightVocoderDiscriminator.Config(),
        mel: LogMelSpectrogram.Config = _default_mel_cfg,
        use_acoustic_features: bool = False,
        weight_mel: float = 45.0,
        weight_feat: float = 1.0,
        weight_adv: float = 1.0,
        weight_f0: float = 30.0,
        lr_decay: float = 0.999,
        betas: List[float] = [0.8, 0.99],
        lr: float = 2e-4,
    ):
        super().__init__()
        self.automatic_optimization = False

        # flag for using data[acoustic_features] instead of mel spectrogram
        self.use_acoustic_features = use_acoustic_features
        self.generator = UltraLightweightVocoderGenerator(**generator)
        self.discriminator = UltraLightweightVocoderDiscriminator(**discriminator)
        self.spectrogram = LogMelSpectrogram(**mel)
        self.weight_mel = weight_mel
        self.weight_adv = weight_adv
        self.weight_feat = weight_feat
        self.weight_f0 = weight_f0
        self.lr_decay = lr_decay
        self.lr = lr
        self.betas = betas

        self.save_hyperparameters()

    def training_step(self, batch: dict):
        real = batch["waveform"]
        f0 = batch.get("f0", None)

        if "acoustic_features" in batch:
            acoustic_features = batch["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(real.sum(1)).detach()

        fake, f0_hat, _, _ = self.generator(acoustic_features, f0)
        self.generator_training_step(real, fake, f0, f0_hat)
        self.discriminator_training_step(real, fake)

    def generator_training_step(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        f0: torch.Tensor,
        f0_hat: torch.Tensor,
    ):
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
        loss_f0 = f0_loss(f0, f0_hat)
        loss_g = (
            loss_mel * self.weight_mel
            + loss_feat * self.weight_feat
            + loss_adv * self.weight_adv
            + loss_f0 * self.weight_f0
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
        self.log("train loss/f0 loss", loss_f0)
        self.log("G", loss_g, prog_bar=True, logger=False)

    def _test_or_validate_batch(self, batch):
        waveform = batch["waveform"]

        if self.use_acoustic_features:
            acoustic_features = batch["acoustic_features"]
        else:
            acoustic_features = self.spectrogram(waveform.sum(1)).detach()

        spec_real = self.spectrogram(waveform).detach()
        fake = self.generator.infer(acoustic_features)
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
