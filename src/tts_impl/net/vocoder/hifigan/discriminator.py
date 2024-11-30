from dataclasses import dataclass, field
from typing import List, Self

from tts_impl.net.vocoder.discriminator import (
    CombinedDiscriminator,
    MultiPeriodDiscriminator,
    MultiResolutionStftDiscriminator,
    MultiScaleDiscriminator,
)
from tts_impl.utils.config import derive_config

_mrsd_default = MultiResolutionStftDiscriminator.Config()
_mrsd_default.resolutions = []


@derive_config
class HifiganDiscriminator(CombinedDiscriminator):
    def __init__(
        self,
        mpd: MultiPeriodDiscriminator.Config = MultiPeriodDiscriminator.Config(),
        msd: MultiScaleDiscriminator.Config = MultiScaleDiscriminator.Config(),
        mrsd: MultiResolutionStftDiscriminator.Config = _mrsd_default,
    ):
        super().__init__()
        self.discriminators.append(MultiPeriodDiscriminator(**mpd))
        self.discriminators.append(MultiScaleDiscriminator(**msd))
        self.discriminators.append(MultiResolutionStftDiscriminator(**mrsd))


__all__ = ["HifiganDiscriminator"]
