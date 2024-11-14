from typing import Optional, Protocol, Tuple

import torch


class LengthRegurator(Protocol):
    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Batched hidden state to be expanded (B, channels, T_text)
            w: Batched token duration (B, T_text)
            x_masks: Mask tensor (B, T_text)
            y_masks: Mask tensor (B, T_feat)
        Returns:
            x: Expanded hidden state (B, channels, T_feat)
        """


class TextEncoder(Protocol):
    """
    Text Encoder
    """

    def forward(
        self, x: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            x: phoneme ids (B, T_text)
            g: speaker embedding (B, embedding_dim)
        Returns:
            x: phoneme embeddings (B, channels, T_text)
            x_mask: 0=masked, 1=not masked (B, 1, T_text)
        """


class VariationalTextEncoder(Protocol):
    """
    Text Encoder for VAE-based models (e.g. VITS)
    """

    def forward(
        self, x: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            x: phoneme ids (B, T_text)
            g: speaker embedding (B, embedding_dim)
        Returns:
            x: phoneme embeddings, follows gaussian distribution (B, channels, T_text)
            mean: mean, (B, channels, T_text)
            logvar: variance, log-scaled (B, channels, T_text)
            x_mask: 0=masked, 1=not masked (B, 1, T_text)
        """


class AcousticFeatureEncoder(Protocol):
    """
    Acoustic Feature Encoder
    """

    def forward(
        self, x: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            x:  (B, in_channels, T_feat)
            g: speaker embedding (B, embedding_dim)
        Returns:
            x: phoneme embeddings (B, out_channels, T_feat)
            x_mask: 0=masked, 1=not masked (B, 1, T_feat)
        """


class VariationalAcousticFeatureEncoder(Protocol):
    """
    Acoustic Feature Encoder for VAE-based models (e.g. VITS)
    """

    def forward(
        self, x: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            x:  (B, in_channels, T_feat)
            g: speaker embedding (B, embedding_dim)
        Returns:
            x: phoneme embeddings, follows gaussian distribution (B, channels, T_feat)
            mean: mean, (B, channels, T_feat)
            logvar: variance, log-scaled (B, channels, T_feat)
            x_mask: 0=masked, 1=not masked (B, 1, T_feat)
        """


class Flow(Protocol):
    """
    Flow model
    """

    def forward(x: torch.Tensor, reverse: bool, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Tensor
            reverse: bool
        Returns:
            x: Tensor, same to input shape
        """
