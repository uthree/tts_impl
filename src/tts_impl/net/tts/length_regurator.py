import torch
import torch.nn.functional as F
from torch import nn as nn

from tts_impl.functional.length_regurator import (
    duplicate_by_duration,
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
    def __init__(self, sigma_scale=1.0, eps=1e-2):
        super().__init__()
        self.sigma_scale = nn.Parameter(
            torch.tensor([sigma_scale], dtype=torch.float32)
        )
        self.eps = eps

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

        B, C, T_text = x.shape
        if x_mask is not None:
            x = x * x_mask

        # 中心位置の計算
        w = w.transpose(1, 2)  # [B, T_text, 1]
        center = torch.cumsum(w, dim=2) - 0.5 * w  # [B, T_text, 1]

        if y_mask is not None:
            max_len = y_mask.shape[2]
        else:
            max_len = int(w.sum(dim=2).max().item())

        pos = torch.arange(max_len, device=x.device, dtype=x.dtype)[None, None, :]

        delta = pos - center  # [B, T_text, T_feat]
        delta = torch.clamp(delta, min=1e-4, max=1e4)  # Nan対策
        sigma = delta * w

        # 1. 中心地に近いと1に近い値を出す関数 (ガウス分布関数)
        weights = torch.exp(-0.5 * sigma.square() * self.sigma_scale)

        # 2. マスク処理 (入力トークンがない部分を Softmax から除外)
        if x_mask is not None:
            weights = weights.masked_fill(x_mask.transpose(1, 2) == 0.0, -1e9)

        # 4. Softmax (内部で max 引かれるので安定)
        weights = F.softmax(weights, dim=2)

        upsampled = torch.matmul(x, weights)

        if y_mask is not None:
            upsampled = upsampled * y_mask

        return upsampled
