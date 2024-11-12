from typing import Optional

import torch
import torch.nn as nn

from tts_impl.functional.length_regurator import (gaussian_upsampling,
                                                  length_regurator)


class LengthRegurator(nn.Module):
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
            x (Tensor): Batched hidden state to be expanded (B, channels, T_text)
            w (Tensor): Batched token duration (B, T_text)
            x_masks (Tensor): Mask tensor (B, T_feats)
            y_masks (Tensor): Mask tensor (B, T_text)
        Returns:
            Tensor: Expanded hidden state (B, channels, T_feat)
        """
        return length_regurator(x, w, x_mask=x_mask, y_mask=y_mask)


class GaussianUpsampling(nn.Module):
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
            x (Tensor): Batched hidden state to be expanded (B, channels, T_text)
            w (Tensor): Batched token duration (B, T_text)
            x_masks (Tensor): Mask tensor (B, T_feats)
            y_masks (Tensor): Mask tensor (B, T_text)
        Returns:
            Tensor: Expanded hidden state (B, channels, T_feat)
        """
        return gaussian_upsampling(
            x, w, x_mask=x_mask, y_mask=y_mask, delta=self.temperature
        )
