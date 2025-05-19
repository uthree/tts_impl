import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.base.state import StatefulModule


class DepthwiseCausalConv(StatefulModule):
    def __init__(self, d_model: int, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=bias)

    def _sequential_forward(self, x, h):
        pass

    def _parallel_forward(self, x, h):
        pass

    def _initial_state(self, x):
        pass
