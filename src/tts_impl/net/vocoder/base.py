import torch
import torch.nn as nn

import lightning as L

from typing import List, Tuple, Optional


class DiffusionVocoder(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self):
        pass


class GanVocoderGenerator(nn.Module):
    def __init__(self):
        self.with_condition: bool = False
        super().__init__()

    def forward(self, x: torch.Tensor, g=Optional[torch.Tensor]) -> torch.Tensor:
        pass


class GanVocoderDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forwrad(self, x) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pass


class GanVocoder(L.lightningModule):
    generator: GanVocoderGenerator
    discriminator: GanVocoderDiscriminator