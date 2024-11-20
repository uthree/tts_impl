"""Maximum path calculation module.

This code is based on https://github.com/jaywalnut310/vits.
"""

from typing import Literal, Optional

import numpy as np
import torch

from .mas_numba import maximum_path_numba

default_mas_alogirhtm = "numba"
available_mas_algorithms = ["numba"]


try:
    from .core import maximum_path_c

    default_mas_alogirhtm = "cython"
    is_cython_avalable = True
    available_mas_algorithms.append("cython")
except ImportError:
    is_cython_avalable = False
    default_mas_algorithm = "numba"
    # warnings.warn(
    #    "Cython version is not available. Fallback to 'EXPERIMETAL' numba version. "
    #    "If you want to use the cython version, please build it as follows: "
    #    "`cd /monotonic_align; python setup.py build_ext --inplace`"
    # )


def simple_function(x):
    return x + 1


try:
    scripted_function = torch.jit.script(simple_function)
    torch_jit_available = True
except Exception as e:
    torch_jit_available = False

if torch_jit_available:
    from .mas_torch_jit import maximum_path_jit1, maximum_path_jit2

    default_mas_alogirhtm = "jit1"
    available_mas_algorithms.append("jit1")
    available_mas_algorithms.append("jit2")

try:
    import triton

    from .mas_triton import maximum_path_triton

    default_mas_algorithm = "triton"
    available_mas_algorithms.append("triton")
except Exception:
    pass


def maximum_path(
    neg_x_ent: torch.Tensor,
    attn_mask: torch.Tensor,
    algorithm: Optional[Literal["numba", "cython", "jit1", "jit2", "triton"]] = None,
) -> torch.Tensor:
    """Calculate maximum path.

    Args:
        neg_x_ent (Tensor): Negative X entropy tensor (B, T_feats, T_text).
        attn_mask (Tensor): Attention mask (B, T_feats, T_text).
        algorithm: (Optional(str)) The type of algorithm. If not specified, the most suitable algorithm will be chosen automatically.

    Returns:
        Tensor: Maximum path tensor (B, T_feats, T_text).

    """
    if algorithm is None:
        algorithm = default_mas_algorithm

    if algorithm not in available_mas_algorithms:
        raise ValueError(f"MAS algorithm {algorithm} is not available.")

    if algorithm == "cython":
        device, dtype = neg_x_ent.device, neg_x_ent.dtype
        neg_x_ent = neg_x_ent.cpu().numpy().astype(np.float32)
        path = np.zeros(neg_x_ent.shape, dtype=np.int32)
        t_t_max = attn_mask.sum(1)[:, 0].cpu().numpy().astype(np.int32)
        t_s_max = attn_mask.sum(2)[:, 0].cpu().numpy().astype(np.int32)
        maximum_path_c(path, neg_x_ent, t_t_max, t_s_max)
        return torch.from_numpy(path).to(device=device, dtype=dtype)
    elif algorithm == "numba":
        device, dtype = neg_x_ent.device, neg_x_ent.dtype
        neg_x_ent = neg_x_ent.cpu().numpy().astype(np.float32)
        path = np.zeros(neg_x_ent.shape, dtype=np.int32)
        t_t_max = attn_mask.sum(1)[:, 0].cpu().numpy().astype(np.int32)
        t_s_max = attn_mask.sum(2)[:, 0].cpu().numpy().astype(np.int32)
        maximum_path_numba(path, neg_x_ent, t_t_max, t_s_max)
        return torch.from_numpy(path).to(device=device, dtype=dtype)
    elif algorithm == "jit1":
        return maximum_path_jit1(neg_x_ent, attn_mask)
    elif algorithm == "jit2":
        return maximum_path_jit2(neg_x_ent, attn_mask)
    elif algorithm == "triton":
        return maximum_path_triton(neg_x_ent, attn_mask)
    else:
        raise ValueError("Invalid algorithm identifier.")


__all__ = ["maximum_path", "default_mas_alogirhtm", "available_mas_algorithms"]