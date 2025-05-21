import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.common.causal_conv import CachedCausalConv
from tts_impl.net.common.normalization import EmaInstanceNorm
from tts_impl.net.common.mingru import mingru_parallel, mingru_sequential
from tts_impl.net.base.stateful import StatefulModule, StatefulModuleSequential
from tts_impl.utils.config import derive_config
from typing import Optional


class GruxLayer(StatefulModule):
    def __init__(
        self,
        d_model: int,
        d_ffn: Optional[int] = None,
        kernel_size: int = 4,
        p_dropout: float = 0.0,
        layer_scale: float = 1.0,
    ):
        super().__init__()
        if d_ffn is None:
            d_ffn = d_model * 3
        with torch.no_grad():
            self.d_model = d_model
            self.kernel_size = kernel_size
            self.d_h_norm = d_model * 2
            self.d_h_conv = d_model * (kernel_size - 1)
            self.d_h_gru = d_model

            self.norm = EmaInstanceNorm(d_model, elementwise_affine=False)
            self.linear_z = nn.Linear(d_model, d_model)
            self.conv_h = CachedCausalConv(d_model, kernel_size=kernel_size)
            self.ffn_in = nn.Linear(d_model, d_ffn)
            self.ffn_gate = nn.Linear(d_model, d_ffn)
            self.ffn_out = nn.Linear(d_ffn, d_model)
            self.ffn_out.weight.normal_(0.0, layer_scale)
            self.ffn_out.bias.zero_()
            self.dropout = nn.Dropout(p=p_dropout)

    def ffn(self, x):
        x = self.dropout(x)
        x = F.silu(self.ffn_in(x)) * self.ffn_gate(x)
        x = self.dropout(x)
        x = self.ffn_out(x)
        return x

    def _parallel_forward(self, x, h):
        res = x
        h_norm, h_conv, h_gru = torch.split(
            h, [self.d_h_norm, self.d_h_conv, self.d_h_gru], dim=2
        )  # unpack hidden state
        x, h_norm = self.norm(x, h_norm)
        x_z = self.linear_z(x)
        x_h, h_conv = self.conv_h(x, h_conv)
        x, h_gru = mingru_parallel(x_z, x_h, h_gru)
        x = self.ffn(x)
        h = torch.cat([h_norm, h_conv, h_gru], dim=2)  # pack hidden state
        x = x + res
        return x, h

    def _sequential_forward(self, x, h):
        res = x
        h_norm, h_conv, h_gru = torch.split(
            h, [self.d_h_norm, self.d_h_conv, self.d_h_gru], dim=2
        )  # unpack hidden state
        x, h_norm = self.norm(x, h_norm)
        x_z = self.linear_z(x)
        x_h, h_conv = self.conv_h(x, h_conv)
        h_gru = mingru_sequential(x_z, x_h, h_gru)
        x = h_gru
        x = self.ffn(x)
        h = torch.cat([h_norm, h_conv, h_gru], dim=2)  # pack hidden state
        x = x + res
        return x, h

    def _initial_state(self, x):
        h_norm = self.norm._initial_state(x)
        h_conv = self.conv_h._initial_state(x)
        h_gru = torch.zeros(
            (x.shape[0], 1, self.d_model), dtype=x.dtype, device=x.device
        )
        h = torch.cat([h_norm, h_conv, h_gru], dim=2)
        return h


@derive_config
class Grux(StatefulModuleSequential):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        kernel_size: int = 4,
        d_ffn: Optional[int] = None,
        p_dropout: float = 0.0,
    ):
        mods = []
        for _ in range(num_layers):
            layer_scale = 1.0 / num_layers
            mods.append(GruxLayer(d_model, d_ffn, kernel_size, p_dropout, layer_scale))
        mods.append(EmaInstanceNorm(d_model, elementwise_affine=True))
        super().__init__(mods)
