import fire
import torch
import torch.utils.data.dataloader
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from omegaconf import OmegaConf
from tts_impl.net.vocoder.hifigan import HifiganLightningModule
from tts_impl.net.vocoder.nsf_hifigan import NsfhifiganLightningModule
from tts_impl.utils.datamodule import AudioDataModule

torch.set_float32_matmul_precision("medium")


def run_training(
    cache_dir: str = "dataset_cache",
    batch_size: int = 2,
    epochs=20,
):

    Model = NsfhifiganLightningModule
    # initialize lightningmodule
    cfg = Model.default_config()

    cfg.discriminator.periods = []
    cfg.discriminator.scales = []
    cfg.discriminator.resolutions = [120, 240, 480]
    cfg.generator.upsample_initial_channels = 256
    cfg.generator.resblock_dilations = [[1, 3, 9]]
    cfg.generator.resblock_kernel_sizes = [3]

    model = Model(**cfg)

    # initialize datamodule
    datamodule = AudioDataModule(root=cache_dir, batch_size=batch_size, num_workers=1)

    # initialize trainer
    trainer = Trainer(
        max_epochs=epochs,
        precision="bf16-mixed",
        callbacks=[RichProgressBar()],
        log_every_n_steps=25,
    )

    # run training.
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    fire.Fire(run_training)
