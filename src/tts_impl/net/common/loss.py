import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.utils.config import derive_config
from tts_impl.transforms import LogMelSpectrogram
from torch import Tensor

@derive_config
class LogMelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int):
        super().__init__()
        self.melspec = LogMelSpectrogram(sample_rate, n_fft, hop_length, n_mels)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(self.melspec(input), self.melspec(target))
    

@derive_config
class MutiscaleMelspectrogramLoss(nn.Module):
    def __init__(self, sample_rate: int, n_fft: list[int], hop_length: list[int], n_mels: list[int]):
        super().__init__()
        self.melspecs = nn.ModuleList([])
        for n, h, m in zip(n_fft, hop_length, n_mels):
            self.melspecs.append(LogMelSpectrogram(sample_rate, n,h,m))

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = 0
        for m in self.melspecs:
            loss += F.l1_loss(m(input), m(target))
        return loss / len(self.melspecs)