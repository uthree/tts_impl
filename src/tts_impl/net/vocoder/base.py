from typing import List, Optional, Protocol, Tuple

import torch


class VocoderGenerator(Protocol):
    def infer_vocoder(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)


class GanVocoderGenerator(VocoderGenerator):
    """
    A Protocol for GAN-Based Vocoders
    """

    def __init___(
        self, in_channels: int, condition_dim: int = 0, out_channels=1, *args, **kwargs
    ):
        """
        in_channels: Number of channels of acoustic features taken as input
        condition_dim: Dimensionality of the condition (e.g. speaker embedding). Set to 0 to ignore the condition.'
        out_channels: The number of channels in the output audio waveform. Usually, this is 1.
        """
        pass

    def forward(
        self,
        features: torch.Tensor,
        condition: Optional[torch.Tensor],
        f0: Optional[torch.Tensor],
        uv: Optional[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        inputs:
            features: [batch_size, channels, num_frames], dtype=float
            condition: [batch_size, 1, condition_dim] dtype=float, optional
            f0: [batch_size, 1, num_frames], dtype=float, optional
            uv: [batch_size, 1, num_frames], dtype=float, optional
        returns:
            waveform: [batch_size, channels, frames*frame_size]
                where: frame_size is the number of samples per frame.
        """
        pass


class OdeVocoderGenerator(VocoderGenerator):
    pass


class GanVocoderDiscriminator(Protocol):
    def forwrad(self, x) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pass


class Vocoder(Protocol):
    pass


class GanVocoder(Vocoder):
    pass


class OdeVocoder(Vocoder):
    pass


__all__ = [
    "GanVocoder",
    "OdeVocoder",
    "OdeVocoderDiscriminator",
    "GanVocoderGenerator",
    "VocoderGenerator",
    "VocoderGenerator",
    "Vocoder",
]
