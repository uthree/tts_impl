import math

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from tts_impl.net.base.stateful import StatefulModule, StatefulModuleSequential
from tts_impl.net.common.mingru import MinGRU
from tts_impl.net.vocoder.ddsp import HomomorphicVocoder
from tts_impl.utils.config import derive_config


@derive_config
class NhvcEncoder(StatefulModule):
    """
    NHVC Encoder, this module encodes phoneme without speaker-specific information, and estimate pitch, noise-gate.
    """

    def __init__(
        self,
        in_channels: int = 80,
        d_model: int = 128,
        n_layers: int = 4,
        n_fft: int = 1024,
        d_phonemes: int = 64,
        n_phonemes: int = 128,
        n_f0_classes: int = 128,
        fmin: float = 20.0,
        fmax: float = 8000.0,
    ):
        super().__init__()
        self.d_phonemes = d_phonemes
        self.n_f0_classes = n_f0_classes
        self.fft_bin = n_fft // 2 + 1
        self.pre = nn.Linear(in_channels, d_model)
        self.fmin = fmin
        self.fmax = fmax
        self.stack = StatefulModuleSequential(
            [MinGRU(d_model) for _ in range(n_layers)]
        )
        self.post = nn.Linear(d_model, d_phonemes + n_f0_classes + self.fft_bin)
        self.to_phoneme_prob = nn.Conv1d(d_phonemes, n_phonemes, 1, bias=False)

    def _initial_state(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.stack._initial_state(x, *args, **kwargs)

    def _parallel_forward(
        self, x: Tensor, h: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        x = self.pre(x)
        x, h_last = self.stack._parallel_forward(x, h, *args, **kwargs)
        x = self.post(x)
        return x, h_last

    def freq2idx(self, freq: Tensor) -> Tensor:
        log_fmin = math.log(self.fmin)
        log_fmax = math.log(self.fmax)
        log_delta_f = log_fmax - log_fmin

        log_freq = torch.log(torch.clamp(freq.float(), min=self.fmin, max=self.fmax))
        idx = torch.ceil(
            (log_freq - log_fmin) / log_delta_f * (self.n_f0_classes - 1)
        ).long()
        return idx

    def idx2freq(self, idx: Tensor) -> Tensor:
        log_fmin = math.log(self.fmin)
        log_fmax = math.log(self.fmax)
        log_delta_f = log_fmax - log_fmin

        log_freq = ((idx.float() / (self.n_f0_classes - 1)) + log_fmin) * log_delta_f
        freq = torch.exp(log_freq)
        return freq

    def decode_f0(self, probs: Tensor, k: int = 2) -> Tensor:
        """
        Args:
            probs: shape=[batch_size, n_f0_classes, n_frames]

        Returns:
            f0: shape=[batch_size, n_frames]
        """
        uv, f0_probs = torch.split(probs, [1, self.n_f0_classes - 1])
        topk_result = torch.topk(f0_probs, k=k, dim=1)
        uv = (uv > 0.0).float().squeeze(1)
        freqs = topk_result.indices.float()
        probs = torch.softmax(topk_result.values, dim=1)
        f0 = (freqs * probs).sum(dim=1) * uv
        return f0

    def f0_loss(self, probs: Tensor, f0) -> Tensor:
        """
        Args:
            probs: shape=[batch_size, n_f0_classes, n_frames]
            f0: shape=[batch_size, n_frames]

        Returns:
            loss: shape=[]
        """
        uv = (f0 > self.fmin).float()
        uv_hat_logits, f0_logits = torch.split(
            probs.float(), [1, self.n_f0_classes - 1], dim=1
        )
        uv_hat = torch.sigmoid(uv_hat_logits.squeeze(1))
        loss_uv = (uv - uv_hat).abs().mean()
        f0_label = self.freq2idx(f0)
        loss_f0 = F.cross_entropy(f0_logits, f0_label)
        return loss_f0 + loss_uv


class NhvcDecoder(StatefulModule):
    def __init__(
            self,        
            d_model: int = 128,
            n_layers: int = 4,
            n_fft: int = 1024,
            d_phonemes: int = 64,
            gin_channels: int=128,
        ):
        self.pre = nn.Linear(d_phonemes, d_model)
        self.stack = StatefulModuleSequential(
            [MinGRU(d_model, d_cond=gin_channels) for _ in range(n_layers)]
        )
        self.fft_bin = n_fft // 2 + 1
        self.post = nn.Linear(d_model, self.fft_bin * 2)
        self.d_phonemes = d_phonemes

    def _initial_state(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.stack._initial_state(x, *args, **kwargs)

    def _parallel_forward(
        self, x: Tensor, h: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        x = self.pre(x)
        x, h_last = self.stack._parallel_forward(x, h, *args, **kwargs)
        x = self.post(x)
        return x, h_last