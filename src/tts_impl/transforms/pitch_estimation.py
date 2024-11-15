from typing import Literal

import torch
import torch.nn as nn

from tts_impl.functional.f0_estimation import estimate_f0


class PitchEstimation(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        frame_size: int,
        algorithm: Literal["fcpe", "harvest", "dio"] = "harvest",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.algorithm = algorithm
        self.frame_size = frame_size

    def forward(self, wf) -> torch.Tensor:
        return estimate_f0(wf, self.sample_rate, self.frame_size, self.algorithm)
