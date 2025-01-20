import lightning as L

from tts_impl.net.vocoder.hifigan.lightning import \
    HifiganDiscriminator as VitsDiscriminator

from .models import VitsGenerator


class VitsLightningModule(L.LightningModule):
    def __init__(self):
        pass
