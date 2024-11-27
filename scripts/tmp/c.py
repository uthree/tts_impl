import fire
import torch
import torch.utils.data.dataloader
from lightning import Trainer
from lightning.pytorch.callbacks import Checkpoint, RichProgressBar
from tts_impl.net.vocoder.hifigan import HifiganLightningModule
from tts_impl.net.vocoder.nsf_hifigan import NsfhifiganLightningModule
from tts_impl.utils.datamodule import AudioDataModule

torch.set_float32_matmul_precision("medium")


from omegaconf import OmegaConf, DictConfig

Model = HifiganLightningModule
cfg = Model.default_config()
print(cfg)
OmegaConf.save(DictConfig(cfg.__dict__), "test.json")