from typing import Optional

import torch
import torch.nn.functional as F


def gaussian_upsampling(
    x: torch.Tensor,
    w: torch.Tensor,
    x_mask: Optional[torch.Tensor] = None,
    y_mask: Optional[torch.Tensor] = None,
    delta: float = 0.1,
) -> torch.Tensor:
    """
    Gaussian upsampling with fixed temperature as in:
    https://arxiv.org/abs/2010.04301

    Args:
        x (Tensor): Batched hidden state to be expanded (B, channels, T_text)
        w (Tensor): Batched token duration (B, T_text)
        x_masks (Tensor): Mask tensor (B, T_text)
        y_masks (Tensor): Mask tensor (B, T_feats)
        delta: (float), Temperature
    Returns:
        Tensor: Expanded hidden state (B, channels, T_feat)
    """
    x = x.transpose(1, 2)

    B = w.size(0)
    device = w.device

    if y_mask is None:
        T_feats = w.sum(dim=1).max().int()
    else:
        T_feats = y_mask.size(-1)
    t = torch.arange(0, T_feats).unsqueeze(0).repeat(B, 1).to(device).float()
    if y_mask is not None:
        t = t * y_mask.float()

    c = w.cumsum(dim=-1) - w / 2
    energy = -1 * delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2
    if x_mask is not None:
        energy = energy.masked_fill(
            ~(x_mask.unsqueeze(1).repeat(1, T_feats, 1)), -float("inf")
        )

    p_attn = torch.softmax(energy, dim=2)  # (B, T_feats, T_text)
    x = torch.matmul(p_attn, x)

    x = x.transpose(1, 2)
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def duplicate_by_duration(
    x: torch.Tensor,
    w: torch.Tensor,
    x_mask: Optional[torch.Tensor] = None,
    y_mask: Optional[torch.Tensor] = None,
):
    """
    duplicate text embedding by duration.

    Args:
        x (Tensor): Batched hidden state to be expanded (B, channels, T_text)
        w (Tensor): Batched token duration (B, T_text)
        x_masks (Tensor): Mask tensor (B, T_text)
        y_masks (Tensor): Mask tensor (B, feats)
    Returns:
        Tensor: Expanded hidden state (B, channels, T_feat)
    """

    w_ceil = torch.ceil(w).unsqueeze(1)
    if x_mask is None:
        x_mask = torch.ones(x.shape[0], 1, x.shape[2])
    if y_mask is None:
        y_mask = sequence_mask(w_ceil.sum(2))
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = generate_path(w_ceil, attn_mask)
    x = torch.matmul(attn.squeeze(1), x.transpose(1, 2)).transpose(1, 2)
    return x
