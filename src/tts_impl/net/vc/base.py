import lightning as L
import torch
import torch.nn as nn

from typing import Protocol

class VoiceConversion(Protocol):
    pass


class GanVoiceConversion(VoiceConversion):
    pass


class VoiceConersionGenerator(Protocol):
    def infer_vc(self, *args, **kwargs) -> torch.Tensor:
        pass


class GanVoiceConersionGenerator(VoiceConersionGenerator):
    def infer_vc(self, *args, **kwargs) -> torch.Tensor:
        pass
