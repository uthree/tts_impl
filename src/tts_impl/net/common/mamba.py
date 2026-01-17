import torch
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.net.base.stateful import StatefulModule
from tts_impl.utils.config import derive_config


class Mamba(StatefulModule):
    def __init__(
        self,
        d_model: int,
        dt_rank: int,
        d_state: int | None = None,
        initial_period: int = 10000,
    ):
        super().__init__()
        if d_state is None:
            d_state = d_model
        self.d_model = d_model
        self.dt_rank = dt_rank
        self.d_state = d_state

        # TODO: 初期化を論文準拠にする
        self.mag_A = nn.Parameter(torch.zeros(1, 1, d_state))
        self.phase_A = nn.Parameter(
            torch.exp(-torch.arange(d_state).float())[None, None, :]
        )

        self.x_proj = nn.Linear(d_model, d_state * 2 + dt_rank)
        self.dt_proj = nn.Linear(dt_rank, d_state)

    def _parallel_forward(
        self, x: torch.Tensor, h: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, delta_inner = self.x_proj(x).split(
            [self.d_state, self.d_state, self.dt_rank], dim=2
        )
        log_delta = self.dt_proj(delta_inner)
