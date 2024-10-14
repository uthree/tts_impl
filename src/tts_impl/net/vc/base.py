import lightning as L
import torch
import torch.nn as nn


class VoiceConversion(L.LightningModule):
    pass


class GanVoiceConversion(VoiceConversion):
    pass


class VoiceConersionGenerator(nn.Module):
    def infer_vc(self, *args, **kwargs) -> torch.Tensor:
        pass


class GanVoiceConersionGenerator(VoiceConersionGenerator):
    def infer_vc(self, *args, **kwargs) -> torch.Tensor:
        pass
