# Code from https://github.com/imdanboy/jets/blob/main/espnet2/gan_tts/jets/alignments.py
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.functional.forced_align import viterbi_decode


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class ForcedAligner(nn.Module):
    """Alignment Learning Framework proposed for parallel TTS models in:
    https://arxiv.org/abs/2108.10447
    """

    def __init__(self, adim, odim):
        super().__init__()
        self.t_conv1 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.t_conv2 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)

        self.f_conv1 = nn.Conv1d(odim, adim, kernel_size=3, padding=1)
        self.f_conv2 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.f_conv3 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)

    def forward(self, text, feats, x_masks=None):
        """
        Args:
            text :Batched text embedding (B, adim, T_texts)
            feats : Batched acoustic feature (B, odim, T_feats)
            x_masks (Tensor): Mask tensor (B, 1, T_text)

        Returns:
            Tensor: log probability of attention matrix (B, T_feats, T_text)
        """

        text = F.relu(self.t_conv1(text))
        text = self.t_conv2(text)
        text = text.transpose(1, 2)

        feats = F.relu(self.f_conv1(feats))
        feats = F.relu(self.f_conv2(feats))
        feats = self.f_conv3(feats)
        feats = feats.transpose(1, 2)

        dist = feats.unsqueeze(2) - text.unsqueeze(1)
        dist = torch.linalg.norm(dist, ord=2, dim=3)
        score = -dist

        if x_masks is not None:
            score = score.masked_fill((x_masks > 0.0), -np.inf)

        log_p_attn = F.log_softmax(score, dim=-1)
        return log_p_attn

    def align(
        self,
        text: torch.Tensor,
        feats: torch.Tensor,
        text_lengths: torch.Tensor,
        feats_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text :Batched text embedding (B, adim, T_texts)
            feats : Batched acoustic feature (B, odim, T_feats)
            x_masks (Tensor): Mask tensor (B, 1, T_text)

        Returns:
            ds: (Tensor) durations (B, T_text)
            loss: (Tensor) alignment loss, ()
        """
        x_masks = sequence_mask(text_lengths, text.shape[2]).unsqueeze(1)
        log_p_attn = self.forward(text, feats, x_masks=x_masks)
        ds, loss = viterbi_decode(log_p_attn, text_lengths, feats_lengths)
        return ds, loss
