from tts_impl.net.vocoder.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator, CombinedDiscriminator
from typing import List

class HifiganDiscriminator(CombinedDiscriminator):
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        periods: List[int] = [2, 3, 5, 7, 11],
    ):
        super().__init__()
        self.discriminators.append(MultiPeriodDiscriminator(periods))
        self.discriminators.append(MultiScaleDiscriminator(scales))

__all__ = [
    'HifiganDiscriminator'
]