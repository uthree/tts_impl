from typing import Tuple

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from tts_impl.net.base.stateful import StatefulModule, StatefulModuleSequential
from tts_impl.net.common.mingru import MinGRU
from tts_impl.utils.config import derive_config
from tts_impl.net.vocoder.ddsp import SubtractiveVocoder
import math

class NhvcLayer(StatefulModule):
    def __init__(self, d_model: int, gin_channels: int = 0):
        super().__init__()
        self.mingru = MinGRU(d_model)
        self.gin_channels = gin_channels
        if gin_channels > 0:
            self.modulator = nn.Linear(gin_channels, d_model * 2)
        self.mid = nn.Linear(d_model, d_model * 2)
        self.post = nn.Linear(d_model, d_model)

    def _initial_state(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.mingru._initial_state(x, *args, **kwargs)

    def _parallel_forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        g: Tensor | None = None,
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        res = x
        x, h_last = self.mingru._parallel_forward(x, h_prev)
        x_0, x_1 = self.mid(x).chunk(2, dim=2)
        x = self.post(F.silu(x_0) * x_1)
        if g is not None and self.gin_channels > 0:
            gamma, beta = self.modulator(g).chunk(2, dim=2)
            x = x * torch.exp(gamma) + beta
        x = x + res
        return x, h_last


@derive_config
class NhvcEncoder(StatefulModule):
    """
    NHVC Encoder, this module encodes phoneme without speaker-specific information, and estimate pitch, noise-gate.
    """

    def __init__(
        self,
        in_channels: int=80,
        d_model: int = 128,
        n_layers: int = 4,
        n_fft: int = 1024,
        dim_phonemes: int = 64,
        n_f0_classes: int = 128,
        fmin: float=20.0,
        fmax: float=8000.0,
    ):
        super().__init__()
        self.dim_phonemes = dim_phonemes
        self.n_f0_classes = n_f0_classes
        self.fft_bin = n_fft // 2 + 1
        self.pre = nn.Linear(in_channels, d_model)
        self.fmin = fmin
        self.fmax = fmax
        self.stack = StatefulModuleSequential(
            [NhvcLayer(d_model) for _ in range(n_layers)]
        )
        self.post = nn.Linear(d_model, dim_phonemes + n_f0_classes + self.fft_bin)

    def _initial_state(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.stack._initial_state(x, *args, **kwargs)

    def _parallel_forward(
        self, x: Tensor, h: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        x = self.pre(x)
        x, h_last = self.stack._parallel_forward(x, h, *args, **kwargs)
        x = self.post(x)
        return x, h_last

    def freq2idx(self, freq: Tensor) -> Tensor:
        log_fmin = math.log(self.fmin)
        log_fmax = math.log(self.fmax)
        log_delta_f = log_fmax - log_fmin

        log_freq = torch.log(torch.clamp(freq.float(), min=self.fmin, max=self.fmax))
        idx = torch.ceil((log_freq - log_fmin) / log_delta_f * (self.n_f0_classes-1)).long()
        return idx
    
    def idx2freq(self, idx: Tensor) -> Tensor:
        log_fmin = math.log(self.fmin)
        log_fmax = math.log(self.fmax)
        log_delta_f = log_fmax - log_fmin

        log_freq = ((idx.float() / (self.n_f0_classes-1)) + log_fmin) * log_delta_f
        freq = torch.exp(log_freq)
        return freq
    
    def decode_f0(self, probs: Tensor, k: int=2) -> Tensor:
        """
        Args:
            probs: shape=[batch_size, n_f0_classes, n_frames]

        Returns:
            f0: shape=[batch_size, n_frames]
        """
        uv, f0_probs = torch.split(probs, [1, self.n_f0_classes-1])
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
        uv_hat_logits, f0_logits = torch.split(probs.float(), [1, self.n_f0_classes-1], dim=1)
        uv_hat = torch.sigmoid(uv_hat_logits.squeeze(1))
        loss_uv = (uv - uv_hat).abs().mean()
        f0_label = self.freq2idx(f0)
        loss_f0 = F.cross_entropy(f0_logits, f0_label)
        return loss_f0 + loss_uv


@derive_config
class NhvcDecoder(StatefulModule):
    """
    NHVC Decoder, this module estimates parameters of subtractive vocoder
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        dim_phonemes: int = 64,
        n_fft: int = 1024,
        dim_periodicity: int = 16,
        gin_channels: int = 128,
    ):
        super().__init__()
        self.fft_bin = n_fft // 2 + 1
        self.dim_phonemes = dim_phonemes
        self.dim_periodicity = dim_periodicity
        self.pre = nn.Linear(dim_phonemes, d_model)
        self.stack = StatefulModuleSequential(
            [NhvcLayer(d_model, gin_channels=gin_channels) for _ in range(n_layers)]
        )
        self.post = nn.Linear(d_model, dim_periodicity + self.fft_bin)

    def _initial_state(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.stack._initial_state(x, *args, **kwargs)

    def _parallel_forward(
        self, x: Tensor, h: Tensor, g: Tensor | None = None,**kwargs
    ) -> Tuple[Tensor, Tensor]:
        x = self.pre(x)
        x, h_last = self.stack._parallel_forward(x, h, g=g, **kwargs)
        x = self.post(x)
        return x, h_last


@derive_config
class ReverbFilterBank(nn.Module):
    def __init__(self, n_speakers: int = 100, size: int = 2048):
        super().__init__()
        self.emb = nn.Embedding(n_speakers, size)

    def forward(self, sid):
        rev = F.normalize(self.emb(sid), dim=1)
        rev[:, 0] = 1.0
        return rev
    

def normalize(x, dim=1, eps=1e-6):
    mu = x.mean(dim=dim, keepdim=True)
    sigma = x.std(dim=dim, keepdim=True) + 1e-6
    x = (x - mu) / sigma
    return x


@derive_config
class NhvcGenerator(nn.Module):
    def __init__(
        self,
        encoder: NhvcEncoder.Config = NhvcEncoder.Config(),
        decoder: NhvcDecoder.Config = NhvcDecoder.Config(),
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
        reverb_filter_bank: ReverbFilterBank.Config = ReverbFilterBank.Config(),
        n_speakers: int = 100,
        gin_channels: int = 128,
        dim_ssl: int = 256,
    ):
        super().__init__()
        self.encoder = NhvcEncoder(**encoder)
        self.decoder = NhvcDecoder(**decoder)
        self.vocoder = SubtractiveVocoder(**vocoder)
        self.reverb_filter_bank = ReverbFilterBank(**reverb_filter_bank)
        self.speaker_embeddings = nn.Embedding(n_speakers, gin_channels)
        self.to_ssl = nn.Linear(self.decoder.dim_phonemes, dim_ssl)

    def forward(self, spec, f0, ssl, sid):
        spec = spec.transpose(1,2)
        g = self.speaker_embeddings(sid)
        enc_output, _ = self.encoder.forward(spec)
        z, f0_logits, noise_filter = torch.split(enc_output, [self.encoder.dim_phonemes, self.encoder.n_f0_classes, self.encoder.fft_bin], dim=2)
        loss_distill = self.distillation_loss(z, ssl)
        z = z.detach()
        z = normalize(z, dim=1)
        loss_f0 = self.encoder.f0_loss(f0_logits.transpose(1, 2), f0)
        dec_output, _ = self.decoder.forward(z, g=g.unsqueeze(1))
        per, env = torch.split(dec_output, [self.decoder.dim_periodicity, self.decoder.fft_bin], dim=2)
        per = per.transpose(1, 2)
        env = env.transpose(1, 2)
        rev = self.reverb_filter_bank.forward(sid)
        wf = self.vocoder.synthesize(f0, per, env, rev).unsqueeze(1)
        return wf, loss_f0, loss_distill

    def distillation_loss(self, z, ssl):
        z = self.to_ssl(z)
        z = z.mT # [N, C, L]
        ssl = ssl.mT # [N, C', L']
        print(z.shape, ssl.shape)
        ssl = F.interpolate(ssl, z.shape[2])
        return F.l1_loss(z, ssl)
        
