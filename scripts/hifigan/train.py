import lightning
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch.utils.data.dataloader
from tts_impl.utils.dataset import AudioDataset
import torch
import fire
from tts_impl.net.vocoder.hifigan import HifiganLightningModule, HifiganLightningModuleConfig
from omegaconf import OmegaConf

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

torch.set_float32_matmul_precision('medium')

def run_training(cache_dir: str = "dataset_cache", batch_size: int = 1, config="config/hifigan/base.yml", epochs=1):
    model = HifiganLightningModule()
    datamodule = AudioDataModule(cache_dir, batch_size)
    trainer = Trainer(max_epochs=epochs, precision="bf16-mixed", callbacks=RichProgressBar())
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    fire.Fire(run_training)