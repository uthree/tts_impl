import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tts_impl.net.common.grux import Grux
from tts_impl.utils.config import derive_config


@derive_config
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        phoneme_embedding_dim: int = 32,
        d_model=128,
        num_layers: int = 4,
        num_f0_classes: int = 128,
        fmin: float = 20.0,
        fmax: float = 8000.0,
    ):
        super().__init__()
        self.num_f0_classes = num_f0_classes
        self.phoneme_embedding_dim = phoneme_embedding_dim
        self.fmin = fmin
        self.fmax = fmax

        self.pre = nn.Conv1d(in_channels, d_model, 1)
        self.grux = Grux(d_model, num_layers)
        self.post = nn.Conv1d(d_model, phoneme_embedding_dim + num_f0_classes, 1)

    def forward(self, x, h_0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x = self.pre(x)
        x = x.transpose(1, 2)
        x, h_last = self.grux(x, h_0)
        x = x.transpose(1, 2)
        x = self.post(x)
        z, f0_logits = torch.split(
            [self.phoneme_embedding_dim, self.num_f0_classes], dim=1
        )
        return z, f0_logits, h_last

    def decode_f0(self, f0_logits: Tensor, k: int = 2) -> Tensor:
        uv, f0_logits = torch.split(f0_logits, [1, self.num_f0_classes - 1], 1)
        uv = (torch.sigmoid(uv) > 0.5).float()
        topk_result = torch.topk(f0_logits, k=k, dim=1)
        indices = topk_result.indices.float()
        probs = torch.softmax(topk_result.values, dim=1)
        log_delta_f0 = math.log(self.fmax) - math.log(self.fmin)
        log_fmin = math.log(self.fmin)
        log_f0 = (indices * probs).sum(dim=1) / (
            self.num_f0_classes - 1
        ) * log_delta_f0 + log_fmin
        f0 = torch.exp(log_f0) * uv
        return f0

    def f0_loss(self, f0_logits: Tensor, f0: Tensor) -> Tuple[Tensor, Tensor]:
        uv = (f0 > self.fmin).float() * (f0 < self.fmax).float()
        uv_hat, f0_logits = torch.split(f0_logits, [1, self.num_f0_classes - 1], 1)
        uv_hat = torch.sigmoid(uv_hat)
        log_delta_f0 = math.log(self.fmax) - math.log(self.fmin)
        log_fmin = math.log(self.fmin)
        log_f0 = torch.log(torch.clamp(f0, self.fmin, self.fmax))
        f0_label = torch.ceil(
            (log_f0 - log_fmin) / log_delta_f0 * (self.num_f0_classes - 2)
        ).long()
        loss_f0 = F.cross_entropy(f0_logits, f0_label)
        loss_uv = F.l1_loss(uv_hat, uv)
        return loss_f0, loss_uv


@derive_config
class Decoder(nn.Module):
    def __init__(
        self,
        phoneme_embedding_dim: int = 32,
        gin_channels: int = 256,
        d_model: int = 128,
        num_layers: int = 4,
        n_fft: int = 1024,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.fft_bin = n_fft // 2 + 1
        self.pre = nn.Conv1d(phoneme_embedding_dim, d_model, 1)
        self.grux = Grux(d_model, num_layers, d_condition=gin_channels)
        self.post = nn.Conv1d(d_model, self.fft_bin * 2, 1)

    def forward(self, x, h_0=None, g=None):
        x = self.pre(x)
        x = x.transpose(1, 2)
        if g is not None:
            g = g.transpose(1, 2)
        x, h_last = self.grux(x, h_0, c=g)
        x = x.transpose(1, 2)
        x = torch.exp(x)
        se, ap = torch.chunk(x, 2, dim=1)
        return se, ap, h_last
