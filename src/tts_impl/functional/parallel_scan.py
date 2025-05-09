import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_scan_log(log_coeffs, log_values):
    """
    Heinsen parallel scan

    Args:
        log_coeffs: shape=(batch_size, seq_len, d_model)
        log_values: shape=(batch_size, seqa_len + 1, d_model)
    
    Returns:
        log_h
    """
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()
