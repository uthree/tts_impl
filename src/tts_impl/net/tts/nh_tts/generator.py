from torch import nn as nn

from tts_impl.utils.config import derive_config


@derive_config
class NhttsGenerator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
