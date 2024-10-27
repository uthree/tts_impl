import lightning as L

from tts_impl.net.vocoder.hifigan import \
    HifiganDiscriminator as VitsDiscriminator

from .models import DurationPredictor
from .models import HifiganGenerator as Decoder
from .models import (PosteriorEncoder, ResidualCouplingBlock,
                     StochasticDurationPredictor, TextEncoder, VitsGenerator)


class Vits(L.LightningModule):
    def __init__(self):
        pass


__all__ = [
    "VitsGenerator",
    "Vits",
    "VitsDiscriminator",
    "TextEncoder",
    "ResidualCouplingBlock",
    "PosteriorEncoder",
    "DurationPredictor",
    "StochasticDurationPredictor",
    "Decoder",
]
