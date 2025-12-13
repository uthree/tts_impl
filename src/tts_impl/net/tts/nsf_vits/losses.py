import torch
from torch import nn as nn


def safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp_min(x, min=1e-8))


def log_f0_loss(
    f0_hat: torch.Tensor, f0: torch.Tensor, uv: torch.Tensor | None = None
) -> torch.Tensor:
    if uv is None:
        uv = (f0 >= 20.0).float()

    log_f0_hat = safe_log(f0_hat.float())
    with torch.no_grad():
        log_f0 = safe_log(f0.float()).detach()

    delta = (log_f0_hat - log_f0).abs()
    loss = (delta * uv).sum() / (uv.sum() + 1e-8)
    return loss
