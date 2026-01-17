import torch


def interpolate_f0(f0: torch.Tensor, f0_min: float = 20.0) -> torch.Tensor:
    """
    Interpolate unvoiced regions using linear interpolation.

    For unvoiced regions (where f0=0.0 or f0 < f0_min, indicating no periodicity
    detected or unreliable pitch), this function fills them with linearly
    interpolated values from the surrounding voiced regions.

    Args:
        f0: Tensor, shape=(N, L) or (L,). F0 values where 0.0 or values below
            f0_min indicate unvoiced.
        f0_min: float, minimum valid F0 value. Values below this threshold are
            treated as unvoiced. Default is 20.0 Hz.

    Returns:
        interpolated_f0: Tensor, same shape as input. Unvoiced regions are
            filled with linearly interpolated values. Leading/trailing unvoiced
            regions are filled with the nearest voiced value.
    """
    if f0.ndim == 1:
        return _interpolate_f0_1d(f0, f0_min)
    elif f0.ndim == 2:
        results = []
        for i in range(f0.shape[0]):
            results.append(_interpolate_f0_1d(f0[i], f0_min))
        return torch.stack(results, dim=0)
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {f0.ndim}D")


def _interpolate_f0_1d(f0: torch.Tensor, f0_min: float = 20.0) -> torch.Tensor:
    """
    Interpolate unvoiced regions for a 1D f0 tensor.

    Args:
        f0: Tensor, shape=(L,). F0 values where values below f0_min indicate unvoiced.
        f0_min: float, minimum valid F0 value.

    Returns:
        interpolated_f0: Tensor, shape=(L,).
    """
    device = f0.device
    dtype = f0.dtype
    f0 = f0.clone()

    # Find voiced (f0 >= f0_min) indices
    voiced_mask = f0 >= f0_min
    voiced_indices = torch.where(voiced_mask)[0]

    # If all unvoiced or all voiced, return as-is
    if len(voiced_indices) == 0:
        return f0
    if voiced_mask.all():
        return f0

    # Get first and last voiced indices
    first_voiced = voiced_indices[0].item()
    last_voiced = voiced_indices[-1].item()

    # Fill leading unvoiced region with first voiced value
    if first_voiced > 0:
        f0[:first_voiced] = f0[first_voiced]

    # Fill trailing unvoiced region with last voiced value
    if last_voiced < len(f0) - 1:
        f0[last_voiced + 1 :] = f0[last_voiced]

    # Find unvoiced regions between voiced regions and interpolate
    unvoiced_mask = f0 < f0_min
    unvoiced_indices = torch.where(unvoiced_mask)[0]

    if len(unvoiced_indices) == 0:
        return f0

    # For each unvoiced index, find surrounding voiced indices and interpolate
    for idx in unvoiced_indices:
        idx_val = idx.item()

        # Find the nearest voiced index before this position
        prev_voiced = voiced_indices[voiced_indices < idx]
        if len(prev_voiced) == 0:
            continue
        prev_idx = prev_voiced[-1].item()

        # Find the nearest voiced index after this position
        next_voiced = voiced_indices[voiced_indices > idx]
        if len(next_voiced) == 0:
            continue
        next_idx = next_voiced[0].item()

        # Linear interpolation
        t = (idx_val - prev_idx) / (next_idx - prev_idx)
        f0[idx_val] = f0[prev_idx] * (1 - t) + f0[next_idx] * t

    return f0
