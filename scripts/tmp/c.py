import torch
import torch.utils.data.dataloader

from tts_impl.net.vocoder.hifigan import HifiganLightningModule

torch.set_float32_matmul_precision("medium")


from omegaconf import DictConfig, OmegaConf

Model = HifiganLightningModule
cfg = Model.default_config()
print(cfg)
OmegaConf.save(DictConfig(cfg.__dict__), "test.json")
