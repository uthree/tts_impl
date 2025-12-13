from tts_impl.net.vocoder.discriminator import (
    CombinedDiscriminator,
    MultiPeriodDiscriminator,
    MultiResolutionStftDiscriminator,
    MultiResolutionXcorrDiscriminator,
    MultiScaleDiscriminator,
)
from tts_impl.utils.config import derive_config

_mrsd_default = MultiResolutionStftDiscriminator.Config()
_mrsd_default.hop_size = []
_mrsd_default.n_fft = []

_mrxd_default = MultiResolutionXcorrDiscriminator.Config()
_mrxd_default.hop_size = []
_mrxd_default.n_fft = []


@derive_config
class HifiganDiscriminator(CombinedDiscriminator):
    def __init__(
        self,
        mpd: MultiPeriodDiscriminator.Config = MultiPeriodDiscriminator.Config(),
        msd: MultiScaleDiscriminator.Config = MultiScaleDiscriminator.Config(),
        mrsd: MultiResolutionStftDiscriminator.Config = _mrsd_default,
        mrxd: MultiResolutionXcorrDiscriminator.Config = _mrxd_default,
    ):
        super().__init__()
        self.discriminators.append(MultiPeriodDiscriminator(**mpd))
        self.discriminators.append(MultiScaleDiscriminator(**msd))
        self.discriminators.append(MultiResolutionStftDiscriminator(**mrsd))
        self.discriminators.append(MultiResolutionXcorrDiscriminator(**mrxd))


__all__ = ["HifiganDiscriminator"]
