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

    def forward(self, waveform) -> torch.Tensor:
        """
        Args:
            waveform: Tensor, shape=[batch_size, channels, length]

        Returns:
            waveform: Tensor, shape=[batch_size, length // frame_size]
        """
        return estimate_f0(waveform, self.sample_rate, self.frame_size, self.algorithm)
