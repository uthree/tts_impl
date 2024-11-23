from typing import List

from tts_impl.net.vocoder.discriminator import (
    CombinedDiscriminator, MultiPeriodDiscriminator,
    MultiResolutionStftDiscriminator, MultiScaleDiscriminator)

from dataclasses import dataclass, field


@dataclass
class HifiganDiscriminatorConfig:
    scales: List[int] = field(default_factory = lambda: [1, 2, 4])
    periods: List[int] = field(default_factory= lambda: [2, 3, 5, 7, 11])
    resolutions: List[int] = field(default_factory= lambda: [])


class HifiganDiscriminator(CombinedDiscriminator):
    def __init__(
        self,
        scales: List[int] = [2, 3, 5, 7, 11],
        periods: List[int] = [1, 2, 4],
        resolutions: List[int] = [],
    ):
        super().__init__()
        self.discriminators.append(MultiPeriodDiscriminator(periods))
        self.discriminators.append(MultiScaleDiscriminator(scales))
        self.discriminators.append(MultiResolutionStftDiscriminator(resolutions))


__all__ = ["HifiganDiscriminator"]
