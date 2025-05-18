import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.functional.midi import freq2note, note2freq
from tts_impl.net.common.convnext import ConvNeXt1d
from tts_impl.net.common.mingru import MinGRU


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, glu: bool = False):
        super().__init__()
        self.glu = glu
        if glu:
            self.linear_in = nn.Linear(
                d_model,
                d_ffn * 2,
            )
        else:
            self.linear_in = nn.Linear(
                d_model,
                d_ffn,
            )
        self.linear_out = nn.Linear(d_ffn, d_model)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        d_model: int = 384,
        num_layers: int = 6,
        phoneme_embedding_dim: int = 64,
        pitch_estimation_bin: int = 128,
        fmin: float = 20.0,
        fmax: float = 4000.0,
    ):
        super().__init__()

        self.net_pre = nn.Linear(in_channels, d_model)
        self.layers = nn.ModuleList([MinGRU(d_model) for _ in range(num_layers)])

        out_channels = phoneme_embedding_dim + pitch_estimation_bin + 2
        self.net_post = nn.Linear(d_model, out_channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pre(x)
        for layer in self.layers:
            x = layer(x)
        x = self.post(x)
        x = x.transpose(1, 2)
        return x
