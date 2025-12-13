import numpy as np
import torch
from numba import jit


@jit(nopython=True)
def _average_by_duration(ds, xs, text_lengths, feats_lengths):
    B = ds.shape[0]
    xs_avg = np.zeros_like(ds)
    ds = ds.astype(np.int32)
    for b in range(B):
        t_text = text_lengths[b]
        t_feats = feats_lengths[b]
        d = ds[b, :t_text]
        d_cumsum = d.cumsum()
        d_cumsum = [0] + list(d_cumsum)
        x = xs[b, :t_feats]
        for n, (start, end) in enumerate(
            zip(d_cumsum[:-1], d_cumsum[1:], strict=False)
        ):
            if len(x[start:end]) != 0:
                xs_avg[b, n] = x[start:end].mean()
            else:
                xs_avg[b, n] = 0
    return xs_avg


def average_by_duration(ds, xs, text_lengths, feats_lengths):
    """
    Args:
        ds (Tensor): Batched token duration (B,T_text)
        xs (Tensor): Batched feature sequences to be averaged (B, T_feats)
        text_lengths (Tensor): Text length tensor (B,)
        feats_lengths (Tensor): Feature length tensor (B,)
    Returns:
        Tensor: Batched feature averaged according to the token duration (B, T_text)
    """
    device = ds.device
    args = [ds, xs, text_lengths, feats_lengths]
    args = [arg.detach().cpu().numpy() for arg in args]
    xs_avg = _average_by_duration(*args)
    xs_avg = torch.from_numpy(xs_avg).to(device)
    return xs_avg
