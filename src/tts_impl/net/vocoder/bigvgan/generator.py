
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm

from typing import Optional, List

from tts_impl.net.vocoder.hifigan.utils import get_padding
from tts_impl.net.vocoder import GanVocoderGenerator

from alias_free_torch.act import Activation1d