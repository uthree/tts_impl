from typing import Optional

import torch
import torch.nn as nn
from tts_impl.functional.length_regurator import (
    duplicate_by_duration,
    gaussian_upsampling,
)
from tts_impl.net.base.tts import LengthRegurator


class DuplicateByDuration(nn.Module, LengthRegurator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Batched hidden state to be expanded (B, channels, T_text)
            w: Batched token duration (B, T_text)
            x_masks: Mask tensor (B, T_feats)
            y_masks: Mask tensor (B, T_text)
        Returns:
            x(Tensor): Expanded hidden state (B, channels, T_feat)
        """
        return duplicate_by_duration(x, w, x_mask=x_mask, y_mask=y_mask)


class GaussianUpsampling(nn.Module, LengthRegurator):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None,
    ):
        """
        Gaussian upsampling with fixed temperature as in:
        https://arxiv.org/abs/2010.04301

        Args:
            x: Batched hidden state to be expanded (B, channels, T_text)
            w: Batched token duration (B, T_text)
            x_masks: Mask tensor (B, T_feats)
            y_masks: Mask tensor (B, T_text)
        Returns:
            x: Expanded hidden state (B, channels, T_feat)
        """
        return gaussian_upsampling(
            x, w, x_mask=x_mask, y_mask=y_mask, delta=self.temperature
        )
