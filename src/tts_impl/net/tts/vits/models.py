import math
from typing import List, Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F
from tts_impl.functional import monotonic_align
from tts_impl.net.base.tts import (
    Invertible,
    LengthRegurator,
    VariationalAcousticFeatureEncoder,
    VariationalTextEncoder,
)
from tts_impl.net.tts.length_regurator import DuplicateByDuration
from tts_impl.net.vocoder.hifigan.lightning import HifiganGenerator
from tts_impl.utils.config import derive_config

from . import attentions, commons, modules


@derive_config
class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 192,
        filter_channels: int = 192,
        kernel_size: int = 5,
        p_dropout: float = 0.1,
        n_flows: int = 4,
        gin_channels: int = 0,
        condition_backward: bool = False,
    ):
        super().__init__()
        # it needs to be removed from future version.
        filter_channels = in_channels
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.condition_backward = condition_backward

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            if not self.condition_backward:
                g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


@derive_config
class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 192,
        filter_channels: int = 256,
        kernel_size: int = 5,
        p_dropout: float = 0.1,
        gin_channels=0,
        condition_backward: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.condition_backward = condition_backward

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            if not self.condition_backward:
                g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


@derive_config
class TextEncoder(nn.Module, VariationalTextEncoder):
    def __init__(
        self,
        n_vocab: int = 256,
        out_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: int = 0.1,
        window_size: Optional[int] = 4,
        glu: bool = False,
        rotary_pos_emb: bool = False,
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        prenorm: bool = False,
        activation: Literal["relu", "gelu", "silu"] = "relu",
        gin_channels: int = 0,
        share_relative_attn_bias: bool = True,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.take_condition = gin_channels > 0

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if self.take_condition:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            norm=norm,
            prenorm=prenorm,
            activation=activation,
            glu=glu,
            rotary_pos_emb=rotary_pos_emb,
            share_relative_attn_bias=share_relative_attn_bias,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

        if g is not None and self.take_condition:
            x += self.cond(g).transpose(1, 2)

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


# det ∂f(x)/∂x = 1
@derive_config
class ResidualCouplingBlock(nn.Module, Invertible):
    def __init__(
        self,
        channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


@derive_config
class PosteriorEncoder(nn.Module, VariationalAcousticFeatureEncoder):
    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


_default_decoder_config = HifiganGenerator.Config()
_default_decoder_config.in_channels = 192


@derive_config
class VitsGenerator(nn.Module):
    def __init__(
        self,
        posterior_encoder: PosteriorEncoder.Config = PosteriorEncoder.Config(),
        text_encoder: TextEncoder.Config = TextEncoder.Config(),
        decoder: HifiganGenerator.Config = _default_decoder_config,
        flow: ResidualCouplingBlock.Config = ResidualCouplingBlock.Config(),
        duration_predictor: DurationPredictor.Config = DurationPredictor.Config(),
        stochastic_duration_predictor: StochasticDurationPredictor.Config = StochasticDurationPredictor.Config(),
        n_speakers: int = 0,
        gin_channels: int = 0,
        use_dp: bool = False,
        use_sdp: bool = True,
        segment_size: int = 32,
        mas_noise: float = 0.0,
    ):
        super().__init__()
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.segment_size = segment_size

        self.use_sdp = use_sdp
        self.use_dp = use_dp

        self.mas_noise = mas_noise

        self.enc_p = TextEncoder(**text_encoder)
        self.dec = HifiganGenerator(**decoder)
        self.enc_q = PosteriorEncoder(**posterior_encoder)
        self.flow = ResidualCouplingBlock(**flow)
        self.dec = HifiganGenerator(**decoder)
        self.lr = DuplicateByDuration()

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

        if self.use_sdp:
            l_length = self.sdp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
            l_length_all += l_length

        if self.use_dp:
            logw_hat = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_hat) ** 2, [1, 2]) / torch.sum(
                x_mask
            )  # for averaging
            l_length_all += l_length
        return l_length_all, logw, logw_hat

    def forward(
        self, x, x_lengths, y, y_lengths, sid=None, w=None
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

        # slice
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)

        outputs = {
            "fake": o,
            "loss_dur": loss_dur,
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

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
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
