import fire
import lightning
import torch
import torch.utils.data.dataloader
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from omegaconf import OmegaConf
from tts_impl.net.vocoder.hifigan import (
    HifiganLightningModule,
    HifiganLightningModuleConfig,
)
from tts_impl.utils.dataset import AudioDataset


# Tentative implementation
class AudioDataModule(LightningDataModule):
    def __init__(self, cache_dir, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.cache_dir = cache_dir

    def setup(self, stage: str):
        self.dataset = AudioDataset(self.cache_dir)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.dataloader.Any:
        return torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True)

    def test_dataloader(self) -> torch.utils.data.dataloader.Any:
        return torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True)


torch.set_float32_matmul_precision("medium")


def run_training(
    cache_dir: str = "dataset_cache",
    batch_size: int = 1,
    epochs=1,
):
    model = HifiganLightningModule(
        discriminator={"periods": []},
        generator={
            "upsample_initial_channels": 256,
            "resblock_kernel_sizes": [3],
            "resblock_dilations": [[1, 3, 9]],
        },
    )
    datamodule = AudioDataModule(cache_dir, batch_size)
    trainer = Trainer(
        max_epochs=epochs,
        precision="bf16-mixed",
        callbacks=[RichProgressBar()],
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    fire.Fire(run_training)
