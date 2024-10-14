
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from alias_free_torch.act import Activation1d
from torch.nn.utils.parametrizations import weight_norm

from tts_impl.net.vocoder import GanVocoderGenerator
from tts_impl.net.vocoder.hifigan.utils import get_padding
