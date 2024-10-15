# based on https://github.com/imdanboy/jets/blob/main/espnet2/gan_tts/jets/length_regulator.py

import torch
from typing import Optional


def gaussian_upsampling(hs: torch.Tensor, ds: torch.Tensor, h_masks:Optional[torch.Tensor]=None, d_masks:Optional[torch.Tensor] = None, delta: int=0.1) -> torch.Tensor:
    """
    Args:
        hs (Tensor): Batched hidden state to be expanded (B, T_text, adim)
        ds (Tensor): Batched token duration (B, T_text)
        h_masks (Tensor): Mask tensor (B,T_feats)
        d_masks (Tensor): Mask tensor (B,T_text)
    Returns:
        Tensor: Expanded hidden state (B, T_feat, adim)
    """
    B = ds.size(0)
    device = ds.device

    if h_masks is None:
        T_feats = ds.sum().int()
    else:
        T_feats = h_masks.size(-1)
    t = torch.arange(0, T_feats).unsqueeze(0).repeat(B, 1).to(device).float()
    if h_masks is not None:
        t = t * h_masks.float()

    c = ds.cumsum(dim=-1) - ds / 2
    energy = -1 * delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2
    if d_masks is not None:
        energy = energy.masked_fill(
            ~(d_masks.unsqueeze(1).repeat(1, T_feats, 1)), -float("inf")
        )

    p_attn = torch.softmax(energy, dim=2)  # (B, T_feats, T_text)
    hs = torch.matmul(p_attn, hs)
    return hs


class GaussianUpsampling(torch.nn.Module):
    """
    Gaussian upsampling with fixed temperature as in:
    https://arxiv.org/abs/2010.04301
    """

    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta

    def forward(self, hs: torch.Tensor, ds: torch.Tensor, h_masks:Optional[torch.Tensor]=None, d_masks:Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hs (Tensor): Batched hidden state to be expanded (B, T_text, adim)
            ds (Tensor): Batched token duration (B, T_text)
            h_masks (Tensor): Mask tensor (B,T_feats)
            d_masks (Tensor): Mask tensor (B,T_text)
        Returns:
            Tensor: Expanded hidden state (B, T_feat, adim)
        """
        return gaussian_upsampling(hs, ds, h_masks, d_masks, self.delta)
