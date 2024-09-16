from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class HifiganMelConfig:
    segment_size: int = 8192
    num_mels: int = 80
    n_fft: int = 1024
    hop_size: int = 256
    sample_rate: int = 22050

@dataclass
class HifiganGeneratorConfig:
    input_channels: int = 80
    upsample_rates: List[int] = [8, 8, 2, 2] 
    upsample_kernel_sizes: List[int] = [16, 16, 4, 4]
    upsample_initial_channels: int = 512
    resblock_type: str = "1"
    resblock_kernel_sizes: List[int] = [3, 7, 11]
    resblock_dilations: List[int] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    output_channels: int = 1

@dataclass
class HifiganTrainConfig:
    num_gpus: int = 1
    batch_size: int = 16
    lr: float = 2e-4
    adam_betas: Tuple[float, float] = (0.8, 0.99)
    lr_decay: float = 0.999

@dataclass
class HifiganDiscriminatorConfig:
    periods: List[int] = [2, 3, 5, 7, 11]
    scales: List[int] = [1, 2, 2]

@dataclass
class HifiganConfig:
    mel: HifiganMelConfig
    train: HifiganTrainConfig
    discriminator: HifiganDiscriminatorConfig
    generator: HifiganGeneratorConfig