import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.aligner.forced_align import ForcedAligner
from tts_impl.net.vocoder.ddsp import SubtractiveVocoder
from tts_impl.net.common.convnext import ConvNeXt1d
from tts_impl.utils.config import derive_config


@derive_config
class NhttsGenerator(nn.Module):
    def __init__(
            self,
        ):
        super().__init__()