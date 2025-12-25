import torch
import torch.nn as nn
import torch.nn.functional as F

from tts_impl.net.common.convnext import ConvNeXt1d
from tts_impl.net.tts.vits.attentions import Encoder
from tts_impl.utils.config import derive_config


@derive_config
class TextEncoder(nn.Module):
    def __init__(
        self,
        n_phonemes: int = 256,
        hidden_channels: int = 192,
        filter_channels=192,
        n_heads: int = 6,
        n_layers: int = 4,
    ):
        self.embedding = nn.Embedding(n_phonemes, hidden_channels)
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size=1,
            window_size=0,  # use grobal attention
            glu=True,  # use SwiGLU
            activation="silu",
            norm="rmsnorm",
            prenorm=True,
            share_relative_attn_bias=False,
        )
