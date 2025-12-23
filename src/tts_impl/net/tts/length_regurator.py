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
            w: Batched token duration (B, T_text)
            x_masks: Mask tensor (B, T_text)
            y_masks: Mask tensor (B, T_feats)
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
        x_mask: torch.Tensor | None = None,
        y_mask: torch.Tensor | None = None,
    ):
        """
        Gaussian upsampling with fixed temperature as in:
        https://arxiv.org/abs/2010.04301

        Args:
            x: Batched hidden state to be expanded (B, channels, T_text)
            w: Batched token duration (B, T_text)
            x_masks: Mask tensor (B, T_text)
            y_masks: Mask tensor (B, T_feats)
        Returns:
            x: Expanded hidden state (B, channels, T_feat)
        """
        return gaussian_upsampling(
            x, w, x_mask=x_mask, y_mask=y_mask, delta=self.temperature
        )


class DifferentiableLengthRegulator(nn.Module, LengthRegurator):
    def __init__(self, sigma_scale=1.0, eps=1e-2):
        super().__init__()
        self.sigma_scale = nn.Parameter(
            torch.tensor([sigma_scale], dtype=torch.float32)
        )
        self.eps = eps

    def forward(self, x, w, x_mask=None, y_mask=None):
        B, C, T_text = x.shape
        if x_mask is not None:
            x = x * x_mask

        # 中心位置の計算
        center = torch.cumsum(w, dim=1) - 0.5 * w

        if y_mask is not None:
            max_len = y_mask.shape[2]
        else:
            max_len = int(w.sum(dim=1).max().item())

        pos = torch.arange(max_len, device=x.device, dtype=x.dtype).view(1, 1, -1)
        delta = pos - center.unsqueeze(-1)
        delta = torch.clamp(delta, min=1e-4, max=1e4)  # Nan対策
        print(w.shape, delta.shape)
        sigma = delta * w.unsqueeze(2)

        # 1. 中心地に近いと1に近い値を出す関数 (ガウス分布関数)
        weights = torch.exp(-0.5 * sigma.square() * self.sigma_scale)

        # 2. マスク処理 (入力トークンがない部分を Softmax から除外)
        if x_mask is not None:
            # x_mask: [B, 1, T_text] -> [B, T_text, 1]
            mask = x_mask.transpose(1, 2)
            weights = weights.masked_fill(mask == 0, -1e9)

        # 4. Softmax (内部で max 引かれるので安定)
        weights = F.softmax(weights, dim=1)

        print(x.shape, weights.shape)
        upsampled = torch.matmul(x, weights)

        if y_mask is not None:
            upsampled = upsampled * y_mask

        return upsampled
