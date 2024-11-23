import torch
import torch.nn as nn
from tts_impl.net.common.convnext import Convnext1d


class VocosGenerator(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, n_fft, hop_length, num_layers, causal
    ):
        pass
