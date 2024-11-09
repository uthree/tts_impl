from typing import Optional

import torch


def gaussian_upsampling(
    hs: torch.Tensor,
    ds: torch.Tensor,
    h_masks: Optional[torch.Tensor] = None,
    d_masks: Optional[torch.Tensor] = None,
    delta: float = 0.1,
) -> torch.Tensor:
    """
    Gaussian upsampling with fixed temperature as in:
    https://arxiv.org/abs/2010.04301

    Args:
        hs (Tensor): Batched hidden state to be expanded (B, adim, T_text)
        ds (Tensor): Batched token duration (B, T_text)
        h_masks (Tensor): Mask tensor (B, T_feats)
        d_masks (Tensor): Mask tensor (B, T_text)
        delta: (float), Temperature
    Returns:
        Tensor: Expanded hidden state (B, adim, T_feat)
    """
    hs = hs.transpose(1, 2)

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

    hs = hs.transpose(1, 2)
    return hs


def upsample_by_duration(
    hs: torch.Tensor,
    ds: torch.Tensor,
    h_masks: Optional[torch.Tensor] = None,
    d_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        hs (Tensor): Batched hidden state to be expanded (B, adim, T_text)
        ds (Tensor): Batched token duration (B, T_text)
        h_masks (Tensor): Mask tensor (B, T_feats)
        d_masks (Tensor): Mask tensor (B, T_text)
        delta: (float), Temperature
    Returns:
        Tensor: Expanded hidden state (B, adim, T_feat)
    """
    return gaussian_upsampling(hs, ds, h_masks, d_masks, delta=0.0)