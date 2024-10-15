from abc import ABC, abstractclassmethod

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from tts_impl.net.vocoder.base import GanVocoderDiscriminator


class TextToSpeech(ABC, L.LightningModule):
    pass


class GanTextToSpeech(TextToSpeech):
    pass


class GanTextToSpeechGenerator(ABC, nn.Module):
    def forward(self, *args, **kwargs):
        pass

    def infer_tts(self, *args, **kwargs) -> torch.Tensor:
        pass
