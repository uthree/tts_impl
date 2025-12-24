import torch
from torch.nn import functional as F


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
    x_mask: torch.Tensor | None = None,
    y_mask: torch.Tensor | None = None,
):
    """
    duplicate text embedding by duration.

    Args:
        x (Tensor): Batched hidden state to be expanded (B, channels, T_text)
        w (Tensor): Batched token duration (B, 1, T_text)
        x_mask (Tensor): Mask tensor (B, 1, T_text)
        y_mask (Tensor): Mask tensor (B, 1, feats)

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
