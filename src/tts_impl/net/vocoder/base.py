from typing import List, Optional, Protocol, Tuple

import torch


class VocoderGenerator(Protocol):
    """
    forward pass for training
    """

    def forward(
        self,
        x: torch.Tensor,
        g=Optional[torch.Tensor],
        f0=Optional[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        pass

    def infer_vocoder(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)


class GanVocoderGenerator(VocoderGenerator):
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
