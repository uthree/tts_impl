from typing import List, Optional, Tuple
from abc import ABC, abstractclassmethod

import lightning as L
import torch
import torch.nn as nn


class VocoderGenerator(ABC, nn.Module):
    def __init__(self):
        self.with_condition: bool = False
        self.requires_f0: bool = False
        super().__init__()

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


class GanVocoderDiscriminator(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def forwrad(self, x) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pass


class GanVocoder(ABC, L.LightningModule):
    generator: GanVocoderGenerator
    discriminator: GanVocoderDiscriminator

    def infer_vocoder(self, *args, **kwargs) -> torch.Tensor:
        return self.generator.vocode(*args, **kwargs)


class ODEVocoder(ABC, L.LightningModule):
    generator: ODEVocoderGenerator

    def infer_vocoder(self, *args, **kwargs) -> torch.Tensor:
        return self.generator.vocode(*args, **kwargs)
