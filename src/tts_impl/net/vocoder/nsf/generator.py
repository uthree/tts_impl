import torch
import torch.nn as nn
import torch.nn.functional as F

from tts_impl.net.vocoder.base import GanVocoderGenerator
from tts_impl.net.vocoder.hifigan.generator import ResBlock1, ResBlock2

from .oscillator import Oscillator


class NsfHifiganGenerator(GanVocoderGenerator):
    pass