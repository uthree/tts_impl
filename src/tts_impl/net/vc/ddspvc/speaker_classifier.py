import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.common.convnext import ConvNeXt1d
from tts_impl.utils.config import derive_config


@derive_config
class SpeakerClassifier(nn.Module):
    def __init__(
        self,
        phone_embedding_dim: int = 64,
        num_speakers: int = 256,
        channels: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        self.convnext = ConvNeXt1d(
            phone_embedding_dim,
            channels,
            channels,
            ffn_channels=channels * 3,
            grn=True,
            glu=True,
            num_layers=num_layers,
        )
        self.post = nn.Conv1d(channels, num_speakers, 1)

    def forward(self, x):
        x = self.convnext(x)
        x = self.post(x)
        return x
