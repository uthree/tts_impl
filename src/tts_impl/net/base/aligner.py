from typing import Optional, Protocol

import torch


class Aligner(Protocol):
    def align(
        self,
        text: torch.Tensor,
        feats: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore
        """
        Args:
            text :Batched text embedding (B, C_text, T_text)
            feats : Batched acoustic feature (B, C_feats, T_feats)

        Returns:
            ds: (Tensor) durations (B, T_text)
        """
