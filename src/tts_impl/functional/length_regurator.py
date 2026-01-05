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


def gaussian_upsampling(x, w, x_mask=None, y_mask=None, sigma_scale=0.2):
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
    delta = delta / torch.clamp(w, min=1e-4)

    # 1. 中心地に近いと1に近い値を出す関数 (ガウス分布関数)
    weights = torch.exp(-0.5 * delta.square() * sigma_scale)

    # 2. マスク処理 (入力トークンがない部分を Softmax から除外)
    if x_mask is not None:
        weights = weights.masked_fill(x_mask.transpose(1, 2) == 0.0, -1e9)

    # 4. Softmax (内部で max 引かれるので安定)
    weights = F.softmax(weights, dim=2)

    upsampled = torch.matmul(x, weights)

    if y_mask is not None:
        upsampled = upsampled * y_mask

    return upsampled
