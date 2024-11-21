from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def adjust_size_1d(
    x: torch.Tensor, size: Union[int, Tuple[int], List[int]]
) -> torch.Tensor:
    """
    Adjust the size of a 3D tensor (B, C, L) to the specified (L).

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, L).
        size (tuple, int): Target size (L).

    Returns:
        torch.Tensor: Resized tensor.
    """
    if size.type == tuple:
        size = size[0]

    if x.shape[2] < size:
        pad_size = size - x.shape[2]
        x = F.pad(x, (0, pad_size))
    if x.shape[2] > size:
        x = x[:, :, size]
    return x


def adjust_size_2d(
    x: torch.Tensor, size: Union[Tuple[int, int], List[int]]
) -> torch.Tensor:
    """
    Adjust the size of a 4D tensor (B, C, H, W) to the specified (H, W).

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        size (tuple): Target size (H, W).

    Returns:
        torch.Tensor: Resized tensor.
    """
    target_h, target_w = size
    h, w = x.shape[2], x.shape[3]

    # Adjust height
    if h < target_h:
        pad_size = target_h - h
        x = F.pad(x, (0, 0, 0, pad_size))  # Pad along height
    elif h > target_h:
        x = x[:, :, :target_h, :]  # Crop along height

    # Adjust width
    if w < target_w:
        pad_size = target_w - w
        x = F.pad(x, (0, pad_size, 0, 0))  # Pad along width
    elif w > target_w:
        x = x[:, :, :, :target_w]  # Crop along width

    return x


def adjust_size_3d(
    x: torch.Tensor, size: Union[Tuple[int, int], List[int]]
) -> torch.Tensor:
    """
    Adjust the size of a 5D tensor (B, C, D, H, W) to the specified (D, H, W).

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        size (tuple): Target size (D, H, W).

    Returns:
        torch.Tensor: Resized tensor.
    """
    target_d, target_h, target_w = size
    d, h, w = x.shape[2], x.shape[3], x.shape[4]

    # Adjust depth
    if d < target_d:
        pad_size = target_d - d
        x = F.pad(x, (0, 0, 0, 0, 0, pad_size))  # Pad along depth
    elif d > target_d:
        x = x[:, :, :target_d, :, :]  # Crop along depth

    # Adjust height
    if h < target_h:
        pad_size = target_h - h
        x = F.pad(x, (0, 0, 0, pad_size, 0, 0))  # Pad along height
    elif h > target_h:
        x = x[:, :, :, :target_h, :]  # Crop along height

    # Adjust width
    if w < target_w:
        pad_size = target_w - w
        x = F.pad(x, (0, pad_size, 0, 0, 0, 0))  # Pad along width
    elif w > target_w:
        x = x[:, :, :, :, :target_w]  # Crop along width

    return x


def adjust_size(
    x: torch.Tensor, size: Union[Tuple[int, int], List[int]]
) -> torch.Tensor:
    """
    Adjust the size of a tensor.

    shape supports:
        2D: (N, L)
        3D: (N, C, L)
        4D: (N, C, H, W)
        5D: (N, C, D, H, W)

    Args:
        x (torch.Tensor)
        size (tuple, int)

    Returns:
        torch.Tensor: Resized tensor.
    """
    if x.ndim == 2:
        return adjust_size_1d(x.unsqueeze(1), size).squeeze(1)
    if x.ndim == 3:
        return adjust_size_1d(x, size)
    if x.ndim == 4:
        return adjust_size_2d(x, size)
    if x.ndim == 5:
        return adjust_size_3d(x, size)
