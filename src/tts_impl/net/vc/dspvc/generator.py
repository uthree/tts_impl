import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.functional.midi import freq2note, note2freq
from tts_impl.net.base.stateful import StatefulModule
from tts_impl.net.common.grux import Grux
from tts_impl.utils.config import derive_config


@derive_config
class Encoder(StatefulModule):
    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        num_layers: int = 4,
        kernel_size: int = 4,
        phoneme_embedding_dim: int = 64,
        fmin: float = 10.0,
        fmax: float = 8000.0,
        num_f0_classes: int = 192,
    ):
        super().__init__()
        self.pre = nn.Linear(in_channels, d_model)
        self.grux = Grux(d_model=d_model, num_layers=num_layers)
        self.to_phone = nn.Linear(d_model, phoneme_embedding_dim)
        self.to_f0 = nn.Linear(d_model, num_f0_classes)

    def forward(self, x):
        pass
