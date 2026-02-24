import math
from typing import Literal

import torch
from mpmath.libmp.gammazeta import gamma_stirling_cache
from torch import nn
from torch.nn import functional as F

from tts_impl.functional import monotonic_align
from tts_impl.net.tts.length_regurator import DuplicateByDuration
from tts_impl.net.tts.vits import commons
from tts_impl.net.tts.vits.models import (
    DurationPredictor,
    PosteriorEncoder,
    ResidualCouplingBlock,
    StochasticDurationPredictor,
    TextEncoder,
)
from tts_impl.net.tts.vits.modules import WN
from tts_impl.net.vocoder.nsf_hifigan import NsfhifiganGenerator
from tts_impl.utils.config import derive_config


@derive_config
class PitchEstimator(nn.Module):
    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 3,
        n_layers: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.wn = WN(
            hidden_channels, kernel_size, 1, n_layers, gin_channels=gin_channels
        )
        self.post = nn.Conv1d(hidden_channels, 2, 1)

    def forward(self, x, x_mask, g=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pre(x)
        x = self.wn(x, x_mask, g=g)
        x = self.post(x)
        log_f0, uv = torch.split(x, [1, 1], dim=1)
        uv = torch.sigmoid(uv)
        f0 = torch.exp(log_f0)
        return f0.squeeze(1), uv.squeeze(1)

    def infer(self, x, x_mask, g=None):
        f0, uv = self.forward(x, x_mask, g=g)
        f0 = f0 * (uv > 0.5).float()
        return f0


def safe_log(x):
    return torch.log(torch.clamp(x, min=1e-4))


def f0_loss(f0_hat, uv_hat, f0):
    uv = (f0 > 20.0).float()
    loss_uv = (uv_hat - f0_hat).abs().mean()
    loss_f0 = ((safe_log(f0_hat) - safe_log(f0)).abs() * uv).sum() / (uv.sum() + 1e-4)
    return loss_f0 + loss_uv


@derive_config
class ModernvitsGenerator(nn.Module):
    def __init__(
        self,
        posterior_encoder: PosteriorEncoder.Config = PosteriorEncoder.Config(),
        text_encoder: TextEncoder.Config = TextEncoder.Config(),
        decoder: NsfhifiganGenerator.Config = NsfhifiganGenerator.Config(),
        flow: ResidualCouplingBlock.Config = ResidualCouplingBlock.Config(),
        duration_predictor: DurationPredictor.Config = DurationPredictor.Config(),
        stochastic_duration_predictor: StochasticDurationPredictor.Config = StochasticDurationPredictor.Config(),
        pitch_estimator: PitchEstimator.Config = PitchEstimator.Config(),
        n_speakers: int = 0,
        gin_channels: int = 0,
        use_dp: bool = False,
        use_sdp: bool = True,
        segment_size: int = 32,
        mas_noise: float = 0.0,
        sample_rate: int = 22050,
    ):
        super().__init__()
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.segment_size = segment_size

        self.use_sdp = use_sdp
        self.use_dp = use_dp

        self.mas_noise = mas_noise
        self.sample_rate = sample_rate

        self.enc_p = TextEncoder(**text_encoder)
        self.dec = NsfhifiganGenerator(**decoder)
        self.enc_q = PosteriorEncoder(**posterior_encoder)
        self.flow = ResidualCouplingBlock(**flow)
        self.pe = PitchEstimator(**pitch_estimator)
        self.lr = DuplicateByDuration()

        if use_sdp:
            self.sdp = StochasticDurationPredictor(**stochastic_duration_predictor)
        if use_dp:
            self.dp = DurationPredictor(**duration_predictor)

        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def maximum_path(self, z_p, m_p, logs_p, x_mask, y_mask):
        # calculate likelihood
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            # [b, 1, t_s]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )
            # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn_mask = attn_mask.squeeze(1).transpose(1, 2)
            neg_cent = neg_cent.transpose(1, 2)

            # with noise
            eps = torch.randn_like(neg_cent) * neg_cent.std()
            neg_cent += eps * self.mas_noise

            # Monotonic Alignment Search
            attn = monotonic_align.maximum_path(neg_cent, attn_mask).detach()
        return attn

    def duration_predictor_loss(self, x, x_mask, w, g=None):
        l_length_all = 0
        logw_hat = None
        logw = torch.log(w + 1e-6) * x_mask

        if self.use_dp:
            logw_hat = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_hat) ** 2, [1, 2]) / torch.sum(
                x_mask
            )  # for averaging
            l_length_all += l_length

        if self.use_sdp:
            x = x.float()
            l_length = self.sdp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
            l_length_all += l_length
        return l_length_all, logw, logw_hat

    def forward(
        self, x, x_lengths, y, y_lengths, f0, sid=None, w=None
    ) -> dict[str, torch.Tensor]:
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        # encode prior
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)

        # encode posterior
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)

        # flow
        z_p = self.flow(z, y_mask, g=g)

        # alignment / duration loss
        if w is None:
            attn = self.maximum_path(z_p, m_p, logs_p, x_mask, y_mask)
            w = attn.sum(2).unsqueeze(1)
        loss_dur, logw, logw_hat = self.duration_predictor_loss(x, x_mask, w, g=g)

        # f0 estimation loss
        f0_hat, uv_hat = self.pe(z, y_mask, g=g)
        loss_f0 = f0_loss(
            f0_hat,
            uv_hat,
            f0,
        )

        # expand prior
        m_p = self.lr(m_p, w, x_mask, y_mask)
        logs_p = self.lr(logs_p, w, x_mask, y_mask)

        # slice
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )

        f0_slice = commons.slice_segments(
            f0.unsqueeze(1), ids_slice, self.segment_size
        ).squeeze(1)
        o = self.dec(z_slice, f0=f0_slice, g=g)

        outputs = {
            "fake": o,
            "loss_dur": loss_dur,
            "loss_f0": loss_f0,
            "ids_slice": ids_slice,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
            "logw": logw,
            "logw_hat": logw_hat,
        }

        return outputs

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=0.667,
        length_scale=1.0,
        noise_scale_w=0.8,
        max_len=None,
        use_sdp=True,
        w=None,
    ):
        # speaker embedding
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        # encode text
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # predict duration
        if w is None:
            if use_sdp:
                logw = self.sdp(
                    x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w
                ).squeeze(1)
            else:
                logw = self.dp(x, x_mask, g=g)
            w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)

        # create mask based on max sequence length
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )

        # expand prior
        m_p = self.lr(m_p, w, x_mask, y_mask)
        logs_p = self.lr(logs_p, w, x_mask, y_mask)
        # re-sample from gaussian distribution.
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # flow
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # synthesize
        f0 = self.pe.infer((z * y_mask)[:, :, :max_len], y_mask, g=g)
        o = self.dec((z * y_mask)[:, :, :max_len], f0=f0, g=g)
        return o

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
