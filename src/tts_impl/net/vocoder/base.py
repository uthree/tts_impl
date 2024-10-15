from typing import List, Optional, Protocol, Tuple

import lightning as L
import torch
import torch.nn as nn


class VocoderGenerator(Protocol):
    """
    forward pass for training
    """

    def forward(
        self, x: torch.Tensor, g=Optional[torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        pass

    def infer_vocoder(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)


class GanVocoderGenerator(VocoderGenerator):
    pass


class ODEVocoderGenerator(VocoderGenerator):
    pass


class GanVocoderDiscriminator(Protocol):
    def forwrad(self, x) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pass


class GanVocoder(Protocol):
    def infer_vocoder(self, *args, **kwargs) -> torch.Tensor:
        pass


class ODEVocoder(Protocol):
    def infer_vocoder(self, *args, **kwargs) -> torch.Tensor:
        pass
