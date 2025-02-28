from typing import Literal, Optional

import numpy as np
import torch

from .mas_naive import maximum_path_naive

available_mas_algorithms = ["naive"]
default_mas_alogirhtm = "naive"


try:
    from .core import maximum_path_c  # type: ignore

    default_mas_alogirhtm = "cython"
    is_cython_avalable = True
    available_mas_algorithms.append("cython")
except ImportError:
    is_cython_avalable = False
    # warnings.warn(
    #    "Cython version is not available. Fallback to 'EXPERIMETAL' numba version. "
    #    "If you want to use the cython version, please build it as follows: "
    #    "`cd /monotonic_align; python setup.py build_ext --inplace`"
    # )


try:
    from .mas_torch_jit import maximum_path_jit1, maximum_path_jit2

    default_mas_alogirhtm = "jit1"
    available_mas_algorithms.append("jit1")
    available_mas_algorithms.append("jit2")
    torch_jit_available = True
except ImportError:
    torch_jit_available = False


try:
    import triton

    from .mas_triton import maximum_path_triton

    default_mas_algorithm = "triton"
    available_mas_algorithms.append("triton")
except ImportError:
    pass


try:
    from .mas_numba import maximum_path_numba

    available_mas_algorithms.append("numba")
    default_mas_alogirhtm = "numba"
except Exception:
    pass


@torch.no_grad()
def _mas_on_cpu(attn, attn_mask, fn):
    attn_mask = attn_mask.transpose(1, 2).contiguous()
    neg_x_ent = attn.transpose(1, 2).contiguous()
    device, dtype = neg_x_ent.device, neg_x_ent.dtype
    neg_x_ent = neg_x_ent.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_x_ent.shape, dtype=np.int32)
    t_t_max = attn_mask.sum(1)[:, 0].cpu().numpy().astype(np.int32)
    t_s_max = attn_mask.sum(2)[:, 0].cpu().numpy().astype(np.int32)
    fn(path, neg_x_ent, t_t_max, t_s_max)
    output = torch.from_numpy(path).to(device=device, dtype=dtype).transpose(1, 2)
    return output


def maximum_path(
    attn: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    algorithm: Optional[
        Literal["naive", "numba", "cython", "jit1", "jit2", "triton"]
    ] = None,
) -> torch.Tensor:
    """Calculate maximum path.

    Args:
        attn (Tensor): Negative X entropy tensor (B, T_text, T_feats). should be T_feats >= T_text
        attn_mask (Tensor): Attention mask (B, T_text, T_feats,).
        algorithm: (str) algorithm type.

    Returns:
        Tensor: Maximum path tensor (B, T_text, T_feats).

    Algorithm Details:
        'naive': naive python implementation with numpy.
        'cython': cython implementation from official VITS implementation.
        'numba': numba implementation.
        'jit1': JIT v1 implementation proposed in super MAS.
        'jit2': JIT v2 implementation proposed in super MAS paper.
        'triton': Triton implementation in super MAS paper.

    References:
        [VITS](https://github.com/jaywalnut310/vits)
        [Super Monotonic Alignment Search](https://github.com/supertone-inc/super-monotonic-align)
    """
    if attn_mask is None:
        attn_mask = torch.ones_like(attn)

    if algorithm is None:
        algorithm = default_mas_alogirhtm

    if algorithm == "naive":
        return _mas_on_cpu(attn, attn_mask, mas_naive)
    elif algorithm == "cython":
        return _mas_on_cpu(attn, attn_mask, maximum_path_c)
    elif algorithm == "numba":
        return _mas_on_cpu(attn, attn_mask, maximum_path_numba)
    elif algorithm == "jit1":
        return maximum_path_jit1(attn, attn_mask)
    elif algorithm == "jit2":
        return maximum_path_jit2(attn, attn_mask)
    elif algorithm == "triton":
        return maximum_path_triton(attn, attn_mask)
    else:
        raise ValueError("Invalid algorithm")


__all__ = ["maximum_path", "default_mas_alogorhtm", "available_mas_algorithms"]
