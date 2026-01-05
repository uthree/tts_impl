import torch
import torch.nn.functional as F
from torch import nn as nn

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
        x_mask: torch.Tensor | None = None,
        y_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            x: Batched hidden state to be expanded (B, channels, T_text)
            w: Batched token duration (B, 1, T_text)
            x_masks: Mask tensor (B, 1, T_text)
            y_masks: Mask tensor (B, 1, T_feats)
        Returns:
            x(Tensor): Expanded hidden state (B, channels, T_feat)
        """
        return duplicate_by_duration(x, w, x_mask=x_mask, y_mask=y_mask)


class GaussianUpsampling(nn.Module, LengthRegurator):
    def __init__(self):
        super().__init__()

    def forward(self, x, w, x_mask=None, y_mask=None):
        """
        Args:
            x: Batched hidden state to be expanded (B, channels, T_text)
            w: Batched token duration (B, 1, T_text)
            x_masks: Mask tensor (B, 1, T_text)
            y_masks: Mask tensor (B, 1, T_feats)
        Returns:
            x(Tensor): Expanded hidden state (B, channels, T_feat)
        """

        upsampled = gaussian_upsampling(
            x,
            w,
            x_mask=x_mask,
            y_mask=y_mask,
        )
        return upsampled
