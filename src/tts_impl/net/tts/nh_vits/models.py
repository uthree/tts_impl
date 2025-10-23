import math
from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F
from tts_impl.functional import monotonic_align
from tts_impl.net.tts.length_regurator import DuplicateByDuration
from tts_impl.net.tts.vits import (
    DurationPredictor,
    PosteriorEncoder,
    ResidualCouplingBlock,
    StochasticDurationPredictor,
    TextEncoder,
    commons,
    modules,
)
from tts_impl.net.vocoder.ddsp import SubtractiveVocoder
from tts_impl.utils.config import derive_config
from .losses import log_f0_loss

@derive_config
class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int=192,
                 n_fft: int= 1024,
                 dim_periodicity: int= 16,
                 hidden_channels: int = 192,
                kernel_size: int = 5,
                dilation_rate: int = 1,
                n_layers: int = 16, gin_channels: int=0):
        self.fft_bin = n_fft // 2 + 1
        self.dim_periodicity = dim_periodicity
        out_channels = self.fft_bin + dim_periodicity + 1
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.wn = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x) * x_mask
        x = self.wn(x, x_mask, g=g)
        x = self.post(x) * x_mask
        log_f0, periodicity, spectral_envelope = torch.split(x, [1, self.dim_periodicity, self.fft_bin], dim=1)
        return torch.exp(log_f0), torch.sigmoid(periodicity), torch.exp(spectral_envelope)
    

@derive_config
class NhvitsGenerator(nn.Module):
    def __init__(
        self,
        posterior_encoder: PosteriorEncoder.Config = PosteriorEncoder.Config(),
        text_encoder: TextEncoder.Config = TextEncoder.Config(),
        flow: ResidualCouplingBlock.Config = ResidualCouplingBlock.Config(),
        duration_predictor: DurationPredictor.Config = DurationPredictor.Config(),
        stochastic_duration_predictor: StochasticDurationPredictor.Config = StochasticDurationPredictor.Config(),
        decoder: Decoder.Config = Decoder.Config(),
        vocoder: SubtractiveVocoder.Config = SubtractiveVocoder.Config(),
        n_speakers: int = 0,
        gin_channels: int = 0,
        use_dp: bool = True,
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
        self.dec = Decoder(**decoder)
        self.enc_q = PosteriorEncoder(**posterior_encoder)
        self.flow = ResidualCouplingBlock(**flow)
        self.vocoder = SubtractiveVocoder(**vocoder)
        self.lr = DuplicateByDuration()

        self.vocoder.sample_rate = self.sample_rate

        if use_sdp:
            self.sdp = StochasticDurationPredictor(**stochastic_duration_predictor)
        if use_dp:
            self.dp = DurationPredictor(**duration_predictor)

        if n_speakers > 1:
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
        w = w.unsqueeze(1)
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

        # encode posterior, flow
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        # alignment / duration loss
        if w is None:
            attn = self.maximum_path(z_p, m_p, logs_p, x_mask, y_mask)
            w = attn.sum(2)
        loss_dur, logw, logw_hat = self.duration_predictor_loss(x, x_mask, w, g=g)

        # expand prior
        m_p = self.lr(m_p, w, x_mask, y_mask)
        logs_p = self.lr(logs_p, w, x_mask, y_mask)


        # calculate uv
        uv = (f0 > 20.0).float()

        # slice
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        f0_slice = commons.slice_segments(
            f0.unsqueeze(1), ids_slice, self.segment_size
        ).squeeze(1)
        uv_slice = commons.slice_segments(
            f0.unsqueeze(1), ids_slice, self.segment_size
        ).squeeze(1)

        # decode
        f0_hat, periodicity, spectral_envelope = self.dec.forward(z_slice, x_mask, g=g)

        # pitch estimation loss
        loss_f0 = log_f0_loss(f0_hat, f0_slice, uv_slice)

        # synthesize
        o = self.vocoder.synthesize(f0_slice, periodicity, spectral_envelope)

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
            "f0": f0,
            "f0_hat": f0_hat,
            "uv": uv,
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
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        if w is None:
            if use_sdp:
                logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
            else:
                logw = self.dp(x, x_mask, g=g)
            w = torch.exp(logw) * x_mask * length_scale

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )

        # expand prior
        m_p = self.lr(m_p, w, x_mask, y_mask)
        logs_p = self.lr(logs_p, w, x_mask, y_mask)

        # sample from gaussian dist.
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale


        # flow
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # decoder
        f0, periodicity, spectral_envelope = self.dec.forward(
            (z * y_mask)[:, :, :max_len], y_mask, g=g
        )

        # vocoder
        o = self.vocoder.synthesize(f0, periodicity, spectral_envelope)
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
