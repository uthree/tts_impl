import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.base.stateful import StatefulModule
from tts_impl.net.common.grux import Grux
from tts_impl.net.vocoder.ddsp.vocoder import SubtractiveVocoder
from tts_impl.utils.config import derive_config


@derive_config
class Encoder(StatefulModule):
    def __init__(
        self,
        in_channels: int = 80,
        d_model: int = 128,
        n_layers: int = 4,
        phoneme_embedding_dim: int = 64,
        fmin: float = 10.0,
        fmax: float = 8000.0,
        num_f0_classes: int = 192,
    ):
        super().__init__()
        self.num_f0_classes = num_f0_classes
        self.phoneme_embedding_dim = phoneme_embedding_dim
        self.fmin = fmin
        self.fmax = fmax

        self.pre = nn.Linear(in_channels, d_model)
        self.grux = Grux(d_model=d_model, num_layers=n_layers)
        self.to_phone = nn.Linear(d_model, phoneme_embedding_dim)
        self.to_f0 = nn.Linear(d_model, num_f0_classes)

    def forward(self, x, h=None):
        x = x.transpose(1, 2)
        x = self.pre(x)
        x, h = self.grux(x)
        phone_emb = self.to_phone(x).transpose(1, 2)
        f0_logits = self.to_f0(x).transpose(1, 2)
        return phone_emb, f0_logits, h

    def f0_loss(self, f0_logits, f0, uv):
        f0 = torch.clamp_min(f0, 1.0)
        f0_logits, uv_logits = torch.split(f0_logits, [self.num_f0_classes - 1, 1], dim=1)
        uv_hat = torch.sigmoid(uv_logits.squeeze(1))
        loss_uv = F.l1_loss(uv_hat, uv)
        log_fmin = math.log(self.fmin)
        log_fmax = math.log(self.fmax)
        delta_log_f0 = log_fmax - log_fmin
        f0_label = torch.ceil(torch.clamp_min((torch.log(f0) - log_fmin) / delta_log_f0 * (self.num_f0_classes - 1), 0)).long()
        loss_f0 = F.cross_entropy(f0_logits, f0_label, ignore_index=0)
        return loss_f0, loss_uv

    def decode_f0(self, f0_logits, k: int = 2):
        f0_logits, uv = torch.split(f0_logits, [self.num_f0_classes - 1, 1], dim=1)
        uv = (torch.sigmoid(uv) > 0.5).float()
        log_fmin = math.log(self.fmin)
        log_fmax = math.log(self.fmax)
        delta_log_f0 = log_fmax - log_fmin
        topk_result = torch.topk(f0_logits, dim=1, k=k)
        indices = topk_result.indices.float()
        values = topk_result.values
        probs = torch.softmax(values, dim=1)
        log_f0 = (indices * probs).sum(dim=1) / (
            self.num_f0_classes - 1
        ) * delta_log_f0 + log_fmin
        f0 = torch.exp(log_f0) * uv
        return f0


@derive_config
class Decoder(StatefulModule):
    def __init__(
        self,
        in_channels: int = 64,
        d_model: int = 128,
        n_layers: int = 4,
        d_speaker: int = 256,
        d_periodicity: int = 8,
        n_mels: int = 80,
    ):
        super().__init__()
        self.pre = nn.Linear(in_channels, d_model)
        self.grux = Grux(d_model=d_model, num_layers=n_layers, d_condition=d_speaker)
        self.to_periodicity = nn.Linear(d_model, d_periodicity)
        self.to_envelope = nn.Linear(d_model, n_mels)

    def forward(self, x, h, c):
        x = self.pre(x.transpose(1, 2))
        x, h = self.grux(x, h, c=c)
        per = self.to_periodicity(x).transpose(1, 2)
        env = self.to_envelope(x).transpose(1,2)
        return per, env, h


@derive_config
class DdspvcGenerator(nn.Module):
    def __init__(
        self,
        n_speaker: int = 256,
        d_speaker: int = 256,
        encoder: Encoder.Config = Encoder.Config(),
        decoder: Decoder.Config = Decoder.Config(),
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
        sample_rate : int = 24000
    ):
        super().__init__()
        self.encoder = Encoder(**encoder)
        self.decoder = Decoder(**decoder)
        self.speaker_embedding = nn.Embedding(n_speaker, d_speaker)
        self.vocoder = SubtractiveVocoder(**vocoder)
        self.sample_rate = sample_rate

    def forward(self, x, f0, sid):
        z, f0_logits, _ = self.encoder(x)
        uv = (f0 > 20.0).float()
        loss_f0, loss_uv = self.encoder.f0_loss(f0_logits, f0, uv)
        g = self.speaker_embedding(sid).unsqueeze(1)
        per, env, _ = self.decoder(z, None, g)
        per = torch.sigmoid(per)
        env = torch.sigmoid(env)
        fake = self.vocoder(f0, per, env)
        fake = fake.unsqueeze(1)
        return fake, z, loss_f0, loss_uv