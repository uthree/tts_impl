from typing import Protocol

import torch
from lightning import LightningModule


class GanVocoderGenerator(Protocol):
    """
    GAN-based vocoder
    """

    def forward(
        self,
        x: torch.Tensor,
        g=torch.Tensor | None,
        f0=torch.Tensor | None,
        uv=torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            x: acoustic feature (batch_size, in_channels, length)
            g: optional, speaker embedding
            f0: optional, fundamental frequency
            uv: optional, unvoiced / voiced flag
        Returns:
            waveform: (batch_size, out_channels, length * frame_size)
        """


class GanVocoderDiscriminator(Protocol):
    """
    GAN-based vocoder discriminator
    """

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            x: waveform (batch_size, in_channels, length)
        Returns:
            logits: list[Tensor]
            fmap: list[Tensor]
        """


class SanVocoderDiscriminator(Protocol):
    """
    SAN-based vocoder discriminator
    purposed in https://arxiv.org/abs/2309.02836
    """

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            x: waveform (batch_size, in_channels, length)
        Returns:
            logits: list[Tensor]
            directions: list[Tensor]
            fmap: list[Tensor]
        """
