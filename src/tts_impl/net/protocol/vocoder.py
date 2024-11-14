from typing import Protocol, Optional, Tuple, List
import torch


class GanVocoderGenerator(Protocol):
    """
    GAN-based vocoder
    """
    def forward(self, x: torch.Tensor, g=Optional[torch.Tensor], f0=Optional[torch.Tensor], uv=Optional[torch.Tensor]) -> torch.Tensor:
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
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: waveform (batch_size, in_channels, length)
        Returns:
            logits: List[Tensor]
            fmap: List[Tensor]
        """


    
class SanVocoderDiscriminator(Protocol):
    """
    SAN-based vocoder discriminator
    purposed in https://arxiv.org/abs/2309.02836
    """
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: waveform (batch_size, in_channels, length)
        Returns:
            logits: List[Tensor]
            directions: List[Tensor]
            fmap: List[Tensor]
        """