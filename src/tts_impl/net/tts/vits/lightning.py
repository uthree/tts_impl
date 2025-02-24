import lightning as L
import torch
from tts_impl.net.vocoder.hifigan import HifiganDiscriminator
from tts_impl.net.vocoder.hifigan.lightning import (
    HifiganDiscriminator as VitsDiscriminator,
)
from tts_impl.transforms import LogMelSpectrogram
from tts_impl.utils.config import derive_config

from .models import VitsGenerator


@derive_config
class VitsLightningModule(L.LightningModule):
    def __init__(
        self,
        generator: VitsGenerator.Config,
        discriminator: HifiganDiscriminator.Config,
        mel: LogMelSpectrogram.Config,
        weight_mel: float = 45.0,
        weight_feat: float = 1.0,
        weight_adv: float = 1.0,
        lr: float = 2e-4,
        lr_decay: float = 0.999,
        betas: list[float] = [0.8, 0.99],
    ):
        super().__init__()
        self.automatic_optimization = False

        self.generator = VitsGenerator(**generator)
        self.discriminator = HifiganDiscriminator(**Discriminator)
        self.spectrogram = LogMelSpectrogram(**mel)

        self.weight_mel = weight_mel
        self.weight_adv = weight_adv
        self.weight_feat = weight_feat

        self.lr_decay = lr_decay
        self.lr = lr
        self.betas = betas

    def _discriminator_training_step(
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
