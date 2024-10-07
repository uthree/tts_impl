import lightning as L
import torch

import torch.nn as nn

class GanTextToSpeech(L.LightningModule):
    pass


class GanTextToSpeechGenerator(nn.Module):
    def infer_text_to_speech(self) -> torch.Tensor:
        pass