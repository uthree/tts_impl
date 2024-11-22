from typing import Literal

import torch


def match_features(
    source: torch.Tensor,
    reference: torch.Tensor,
    k: int = 4,
    metrics: Literal["IP", "L2", "cos"] = "L2",
):
    """
    search similar k features and take mean them.

    Args:
        source: (batch_size, channels, length)
        reference: (batch_size, channels, num_reference_tokens)
    Returns:
        result: (batch_size, channels, length)
    """
    with torch.no_grad():
        source = source.transpose(1, 2)
        reference = reference.transpose(1, 2)
        if metrics == "IP":
            sims = torch.bmm(source, reference.transpose(1, 2))
        elif metrics == "L2":
            sims = -torch.cdist(source, reference)
        elif metrics == "cos":
            reference_norm = torch.norm(reference, dim=2, keepdim=True, p=2) + 1e-6
            source_norm = torch.norm(source, dim=2, keepdim=True, p=2) + 1e-6
            sims = torch.bmm(
                source / source_norm, (reference / reference_norm).transpose(1, 2)
            )
        best = torch.topk(sims, k, dim=2)

        result = torch.stack(
            [reference[n][best.indices[n]] for n in range(source.shape[0])], dim=0
        )

    result = result.mean(dim=2)
    result = result.transpose(1, 2)

    return result
