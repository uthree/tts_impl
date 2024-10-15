from typing import Protocol

import lightning as L
import torch
import torch.nn as nn


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
