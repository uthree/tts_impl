import os
import json

import torch
import torch.nn.functional as F


def convert_pad_shape(pad_shape):
	l = pad_shape[::-1]
	pad_shape = [item for sublist in l for item in sublist]
	return pad_shape


def sequence_mask(length, max_length=None):
	if max_length is None:
		max_length = length.max()
	x = torch.arange(max_length, dtype=length.dtype, device=length.device)
	return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
	path = path.unsqueeze(1).transpose(2,3) * mask
	return path


def regurate_length(x: torch.Tensor, x_mask: torch.Tensor, duration: torch.Tensor):
	"""
	x: [b, c, t_x]
	x_mask: [b, c, t_x]
	durations: [b, 1, c]
	"""

	attn = generate_path(duration, x_mask)
	y = torch.matmul(attn.squeeze(1), x.transpose(1, 2)).transpose(1, 2)
	return y