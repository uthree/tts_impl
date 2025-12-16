import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn

from tts_impl.net.base.tts import LengthRegurator
from tts_impl.utils.config import derive_config


class DifferentiableLengthRegulator(nn.Module):
    def __init__(self, sigma_scale=1.0, eps=1e-8):
        super().__init__()
        self.sigma_scale = nn.Parameter(
            torch.tensor([sigma_scale], dtype=torch.float32)
        )
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        y_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Batched hidden state (B, C, T_text)
            w: Batched token duration (B, T_text) ... log domainではなく線形(正の値)であること
        """
        B, C, T_text = x.shape

        # xをマスクしておく
        if x_mask is not None:
            x = x * x_mask

        # 累積和によって各ガウス分布の中心位置を計算
        center = torch.cumsum(w, dim=1) - 0.5 * w  # [B, T_text]

        # 最大長の推定、y_maskがあればその長さを使用
        if y_mask is not None:
            max_len = y_mask.shape[2]
        else:
            max_len = int(w.sum(dim=1).max().item())

        # 位置グリッドの生成
        pos = torch.arange(max_len, device=x.device, dtype=x.dtype).view(1, 1, -1)

        # 距離計算 (SDF equivalent)
        # center: [B, T_text, 1] に拡張してbloadcast
        mu = pos - center.unsqueeze(-1)  # [B, T_text, T_feat]

        # sigma計算
        sigma = w.unsqueeze(-1) * self.sigma_scale.view(1, 1, 1)

        # ガウス分布を計算する
        numerator = -0.5 * mu.square()
        denominator = sigma.square() + self.eps
        weights = torch.exp(numerator / denominator)  # [B, T_text, T_feat]

        # 重みの合計が1になるように正規化
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)

        # アップサンプリング
        upsampled = torch.matmul(x, weights)  # [B, C, T_feat]

        if y_mask is not None:
            # マスクをかける
            upsampled = upsampled * y_mask

        return upsampled
